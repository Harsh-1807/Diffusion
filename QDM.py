# -*- coding: utf-8 -*-

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from Dataset import ClimateDataset
from Network import DiffusionUNet



LR_FILES = [
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/LR_DATA_32/huss_rcm_8km_daily_data_1995-2014_32km.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/LR_DATA_32/mslp_rcm_8km_daily_data_1995-2014_32km.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/LR_DATA_32/tas_rcm_8km_daily_data_1995-2014_32km.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/LR_DATA_32/precip_rcm_8km_daily_data_1995-2014_32km.nc",
]

HR_FILES = [
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/huss_rcm_8km_daily_data_1995-2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/mslp_rcm_8km_daily_data_1995-2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/tas_rcm_8km_daily_data_1995-2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/precip_rcm_8km_daily_data_1995-2014.nc",
]

LOG_FILE  = "corrdiff_v8.log"
SAVE_DIR  = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/CorrDiff/version13/"

_RHO        = 5.0
PRECIP_CH   = 3
N_CH        = 4
N_QUANTILES = 500

USE_PAPER_EMBEDDING = True
SNR_CLIP            = 5.0

# Singapore Gaussian center (lat/lon degrees)
SG_LAT_C  = 1.3
SG_LON_C  = 103.8
SG_SIGMA  = 1.2
SG_WEIGHT = 30.0

# =============================================================================
# LOSS WEIGHTS  (v7: 3 terms, no competition)
# =============================================================================



# =============================================================================
# DDP
# =============================================================================

def setup_ddp():
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


_RANK = 0

def log(msg):
    if _RANK == 0:
        print(msg)
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")


# =============================================================================
# EDM UTILITIES  (unchanged)
# =============================================================================

def get_sigma_schedule(n_steps, device, sigma_min, sigma_max, rho):
    steps  = torch.arange(n_steps + 1, device=device)
    t      = steps / n_steps
    sigmas = (sigma_max**(1/rho) + t*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    return sigmas.clamp(min=0.0)


def edm_precond(sigma, sigma_data):
    sigma   = sigma.reshape(-1, 1, 1, 1)
    c_skip  = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out   = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in    = 1.0 / (sigma_data**2 + sigma**2).sqrt()
    c_noise = sigma.flatten().log() / 4.0
    return c_skip, c_out, c_in, c_noise


def min_snr_weight(sigma, sigma_data, snr_clip=SNR_CLIP):
    snr    = (sigma_data / sigma.clamp(min=1e-8)) ** 2
    weight = (snr_clip / snr).clamp(max=1.0)
    return weight.detach()


# =============================================================================
# EMBEDDING FRAMEWORK  (Aich et al. 2026)
# FIX 3: _determine_noise_steps() replaced with safe bounded version
# =============================================================================

def _cosine_betas(n_steps, device):
    t      = torch.linspace(0, 1, n_steps + 1, device=device)
    alphas = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas = alphas / alphas[0]
    betas  = (1 - alphas[1:] / alphas[:-1]).clamp(max=0.999)
    return betas

def _weak_linear_betas(n_steps, device, beta_start=0.01, beta_end=0.15):
    betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
    return betas


def _apply_noise_steps(field, betas, n_steps):
    # Always run in float32 regardless of autocast context
    x = field.clone().float()
    for t in range(min(n_steps, len(betas))):
        noise = torch.randn_like(x)
        x = (1.0 - betas[t]).sqrt() * x + betas[t].sqrt() * noise
    return x.to(field.dtype)


class EmbeddingFramework:

    def __init__(self, s_freq, n_noise_steps=8, scale_factor=4, device='cpu'):
        self.s_freq        = s_freq
        self.n_noise_steps = n_noise_steps
        self.scale_factor  = scale_factor
        self.device        = device
        self.betas = _weak_linear_betas(n_noise_steps, device,
                                        beta_start=0.01, beta_end=0.10)
        self.qdm_lr        = [None] * N_CH
        self.qdm_hr        = [None] * N_CH
        self._fitted       = False

    def fit_qdm(self, lr_data, hr_data, n_quantiles=N_QUANTILES):
        """
        Fit QDM on NATIVE LR grid vs HR grid.
        NO wet-day filtering during fitting -- fit on full distribution.
        lr_data: [B, C, H_lr, W_lr]
        hr_data: [B, C, H_hr, W_hr]
        """
        log("  Fitting QDM on native LR grid (full distribution) ...")
        q_levels = np.linspace(0, 1, n_quantiles)

        for ch in range(N_CH):
            lr_ch = lr_data[:, ch].reshape(-1).float().cpu().numpy()
            hr_ch = hr_data[:, ch].reshape(-1).float().cpu().numpy()
            self.qdm_lr[ch] = np.quantile(lr_ch, q_levels)
            self.qdm_hr[ch] = np.quantile(hr_ch, q_levels)

        self._fitted = True
        log("  QDM fitting complete.")

    def _qdm_apply_multiplicative(self, arr_np, q_lr, q_hr):
        """Multiplicative QDM for precipitation (Cannon et al. 2015)."""
        shape  = arr_np.shape
        x      = arr_np.ravel().astype(np.float64)
        ranks  = np.linspace(0, 1, len(q_lr))

        # get quantile rank of each value
        tau         = np.interp(x, q_lr, ranks)
        q_model_tau = np.interp(tau, ranks, q_lr)
        q_obs_tau   = np.interp(tau, ranks, q_hr)

        # multiplicative delta -- safe divide
        delta  = x / (np.abs(q_model_tau) + 1e-8)
        result = q_obs_tau * delta
        return result.astype(np.float32).reshape(shape)

    def _qdm_apply_additive(self, arr_np, q_lr, q_hr):
        """Additive QDM for non-precipitation variables."""
        shape  = arr_np.shape
        x      = arr_np.ravel().astype(np.float64)
        ranks  = np.linspace(0, 1, len(q_lr))

        tau         = np.interp(x, q_lr, ranks)
        q_model_tau = np.interp(tau, ranks, q_lr)
        q_obs_tau   = np.interp(tau, ranks, q_hr)

        result = q_obs_tau + (x - q_model_tau)
        return result.astype(np.float32).reshape(shape)

    def f(self, hr_field):
        """
        f(OBS): downsample HR -> upsample back -> add noise.
        Used as condition DURING TRAINING.
        Runs in float32 regardless of autocast.
        """
        with torch.autocast('cuda', enabled=False):
            hr_f32 = hr_field.float()
            B, C, H, W = hr_f32.shape
            sf = self.scale_factor
            H_lr, W_lr = H // sf, W // sf

            hr_ds = F.interpolate(hr_f32, size=(H_lr, W_lr),
                                  mode='bilinear', align_corners=False)
            hr_us = F.interpolate(hr_ds, size=(H, W),
                                  mode='bilinear', align_corners=False)

            betas = self.betas.to(hr_f32.device)
            out = _apply_noise_steps(hr_us, betas, self.n_noise_steps)
        return out.to(hr_field.dtype)

    def g(self, lr_field, hr_size, weight_map=None):
        """
        g(ESM): QDM on native LR grid -> bilinear upsample -> add noise.
        Used as condition DURING INFERENCE.
        Runs in float32 regardless of autocast.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_qdm() first.")

        with torch.autocast('cuda', enabled=False):
            lr_f32 = lr_field.float()
            B, C   = lr_f32.shape[:2]
            H_hr, W_hr = hr_size

            # Step 1: QDM on native LR grid
            lr_qdm = torch.empty_like(lr_f32)
            for ch in range(C):
                arr = lr_f32[:, ch].cpu().numpy()
                # Additive QDM for ALL channels
                # data is log1p+normalized so all channels including
                # precip can be negative -- multiplicative QDM is wrong here
                mapped = self._qdm_apply_additive(
                    arr, self.qdm_lr[ch], self.qdm_hr[ch])
                lr_qdm[:, ch] = torch.tensor(
                    mapped, dtype=torch.float32, device=lr_field.device)

            # Step 2: Bilinear upsample to HR
            lr_up = F.interpolate(lr_qdm, size=(H_hr, W_hr),
                                  mode='bilinear', align_corners=False)

            # Step 3: Same cosine noise as f()
            betas = self.betas.to(lr_field.device)
            out = _apply_noise_steps(lr_up, betas, self.n_noise_steps)

        return out.to(lr_field.dtype)

    def state_dict(self):
        return {"s_freq": self.s_freq, "n_noise_steps": self.n_noise_steps,
                "scale_factor": self.scale_factor,
                "qdm_lr": self.qdm_lr, "qdm_hr": self.qdm_hr,
                "fitted": self._fitted}

    def load_state_dict(self, d):
        self.s_freq        = d["s_freq"]
        self.n_noise_steps = d["n_noise_steps"]
        self.scale_factor  = d["scale_factor"]
        self.qdm_lr        = d["qdm_lr"]
        self.qdm_hr        = d["qdm_hr"]
        self._fitted       = d["fitted"]
        # NOTE: betas rebuilt on self.device -- call .to(device) after this
        #self.betas = _cosine_betas(self.n_noise_steps, torch.device('cpu'))
        self.betas = _weak_linear_betas(self.n_noise_steps, torch.device('cpu'),
                                        beta_start=0.01, beta_end=0.10)

# =============================================================================
# PSD INTERSECTION  (unchanged)
# =============================================================================

def find_psd_intersection(train_loader, device, n_batches=80, qdm_fw=None):
    log("  Computing PSD intersection scale s ...")
    psd_hr_acc = None
    psd_lr_acc = None
    count      = 0

    with torch.no_grad():
        for i, (lr_b, hr_b) in enumerate(train_loader):
            if i >= n_batches:
                break
            hr_p = hr_b[:, PRECIP_CH].to(device)
            lr_p = lr_b[:, PRECIP_CH]        # keep on CPU for QDM

            H, W = hr_p.shape[-2], hr_p.shape[-1]

            if qdm_fw is not None and qdm_fw._fitted:
                arr = lr_p.float().numpy()
                mapped = qdm_fw._qdm_apply_additive(
                    arr, qdm_fw.qdm_lr[PRECIP_CH], qdm_fw.qdm_hr[PRECIP_CH])
                lr_p = torch.tensor(mapped, dtype=torch.float32)

            lr_p = lr_p.to(device)
            lr_up = F.interpolate(lr_p.unsqueeze(1), size=(H, W),
                                  mode="bilinear", align_corners=False).squeeze(1)

            psd_hr = torch.fft.rfft2(hr_p).abs().pow(2).mean(0)
            psd_lr = torch.fft.rfft2(lr_up).abs().pow(2).mean(0)
            psd_hr_acc = psd_hr if psd_hr_acc is None else psd_hr_acc + psd_hr
            psd_lr_acc = psd_lr if psd_lr_acc is None else psd_lr_acc + psd_lr
            count += 1

    

    psd_hr_acc /= count
    psd_lr_acc /= count

    H_full = psd_hr_acc.shape[0]
    W_half = psd_hr_acc.shape[1]
    W_full = (W_half - 1) * 2
    fy     = torch.fft.fftfreq(H_full, device=device).abs()
    fx     = torch.fft.rfftfreq(W_full, device=device).abs()
    freq   = (fy[:, None]**2 + fx[None, :]**2).sqrt()

    eps      = 1e-12
    psd_hr_n = torch.log1p(psd_hr_acc / (psd_hr_acc.mean() + eps))
    psd_lr_n = torch.log1p(psd_lr_acc / (psd_lr_acc.mean() + eps))

    n_bins     = 64
    freq_edges = torch.linspace(0, 0.5, n_bins + 1, device=device)
    hr_1d      = torch.zeros(n_bins, device=device)
    lr_1d      = torch.zeros(n_bins, device=device)

    for b in range(n_bins):
        mask = (freq >= freq_edges[b]) & (freq < freq_edges[b + 1])
        if mask.sum() > 0:
            hr_1d[b] = psd_hr_n[mask].mean()
            lr_1d[b] = psd_lr_n[mask].mean()

    diff       = hr_1d - lr_1d
    cross_idxs = (diff[1:] > 0.005).nonzero(as_tuple=True)[0]

    if len(cross_idxs) == 0:
        s_freq = 0.25
        log(f"  No PSD crossing. Using default s={s_freq:.3f}")
    else:
        s_freq = max(freq_edges[cross_idxs[0] + 1].item(), 0.04)

    log(f"  s_freq = {s_freq:.4f}")
    return s_freq


# =============================================================================
# NETWORK WRAPPER  (unchanged)
# =============================================================================

class SingaporeCorrDiffNet(nn.Module):
    # FIX 1: in_channels now 8 (4 LR-QDM channels), not 12.
    # Old: [residual_noisy(4), mu(4), cond_ch(4)] = 12 channels
    # New: [y_noisy(4), cond_ch(4)] = 8 channels (no mu, matches paper)
    def __init__(self, in_channels=8):
        super().__init__()
        self.net = DiffusionUNet(in_channels=in_channels, out_channels=N_CH)

    def forward(self, x, t):
        H_in, W_in = x.shape[-2], x.shape[-1]
        out = self.net(x, t)
        if out.shape[-2:] != (H_in, W_in):
            out = F.interpolate(out, size=(H_in, W_in),
                                mode="bilinear", align_corners=False)
        return out


# =============================================================================
# MODEL  (FIX 1 applied here -- conditioning path corrected)
# =============================================================================

class SingaporeCorrDiffModel(nn.Module):
    def __init__(self, diffusion_net, sigma_data, sigma_min, sigma_max, rho=_RHO):
        super().__init__()
        self.diffusion_net = diffusion_net
        self.sigma_data    = sigma_data
        self.sigma_min     = sigma_min
        self.sigma_max     = sigma_max
        self.rho           = rho
        # Regressor removed. Model is now fully generative (matches paper).

    def forward_train(self, y_fine, lr, sigma, embedding_fw=None, weight_map=None):
        H, W = y_fine.shape[-2], y_fine.shape[-1]
    
        if embedding_fw is not None and USE_PAPER_EMBEDDING:
            cond_ch = embedding_fw.f(y_fine)
        else:
            cond_ch = F.interpolate(lr, size=(H, W),
                                    mode='bilinear', align_corners=False)
    
        if self.training:
            B = y_fine.size(0)
            drop_mask = 0.95 + 0.05 * torch.rand(B, 1, 1, 1, device=y_fine.device)
            cond_ch = cond_ch * drop_mask
    
        sigma_r = sigma.reshape(-1, 1, 1, 1)
        noise   = torch.randn_like(y_fine)
        y_noisy = y_fine + noise * sigma_r
    
        net_input = torch.cat([y_noisy, cond_ch], dim=1)
    
        c_skip, c_out, c_in, c_noise = edm_precond(sigma, self.sigma_data)
    
        # Network predicts F_x, the denoiser output before skip connection
        F_x = self.diffusion_net(c_in * net_input, c_noise)
    
        # EDM target: network should predict (y_fine - c_skip * y_noisy) / c_out
        # i.e. the residual that when combined with skip gives back y_fine
        target_Fx = (y_fine - c_skip * y_noisy) / (c_out + 1e-8)
    
        return F_x, target_Fx, y_noisy, noise, sigma_r
        
    
        return F_x, target, y_noisy

    @torch.no_grad()
    def sample(self, lr, n_steps=20, embedding_fw=None, weight_map=None):
        H_hr = lr.shape[-2] * 4
        W_hr = lr.shape[-1] * 4

        if embedding_fw is not None and USE_PAPER_EMBEDDING:
            cond_ch = embedding_fw.g(lr, hr_size=(H_hr, W_hr), weight_map=weight_map)
        else:
            cond_ch = F.interpolate(lr, size=(H_hr, W_hr),
                                    mode="bilinear", align_corners=False)

        sigmas = get_sigma_schedule(n_steps, lr.device,
                                     self.sigma_min, self.sigma_max, self.rho)
        x = torch.randn(lr.size(0), N_CH, H_hr, W_hr, device=lr.device) * sigmas[0]

        def denoise(x_in, sig):
            sig_b          = sig.expand(lr.size(0))
            inp            = torch.cat([x_in, cond_ch], dim=1)
            cs, co, ci, cn = edm_precond(sig_b, self.sigma_data)
            Fx             = self.diffusion_net(ci * inp, cn)
            return cs * x_in + co * Fx

        for i in range(n_steps):
            sc = sigmas[i].clamp(min=0.02)
            sn = sigmas[i + 1].clamp(min=0.02)
            Dx = denoise(x, sc)
            d  = (x - Dx) / sc
            if i == n_steps - 1:
                x = Dx
            else:
                xn  = x + (sn - sc) * d
                Dx2 = denoise(xn, sn)
                d2  = (xn - Dx2) / sn
                x   = x + (sn - sc) * (d + d2) * 0.5

        return x


# =============================================================================
# SIGMA_DATA ESTIMATION  (simplified -- no regressor)
# =============================================================================

def estimate_sigma_data(train_loader, device, n_batches=50):
    stds = []
    with torch.no_grad():
        for i, (lr_b, hr_b) in enumerate(train_loader):
            if i >= n_batches:
                break
            hr_b = hr_b.to(device)
            p_max = hr_b[:, PRECIP_CH].reshape(hr_b.size(0), -1).max(dim=1).values
            wet   = p_max > 0.0
            if wet.sum() == 0:
                continue
            stds.append(hr_b[wet, PRECIP_CH].std().item())
    if len(stds) == 0:
        return 0.5
    return float(torch.tensor(stds).mean())


# =============================================================================
# SINGAPORE WEIGHT MAP  (unchanged)
# =============================================================================

def build_singapore_weight_map(H, W, lat, lon, device):
    lat_t = torch.tensor(lat, device=device, dtype=torch.float32)
    lon_t = torch.tensor(lon, device=device, dtype=torch.float32)

    lat_grid = lat_t[:, None].expand(-1, W)
    lon_grid = lon_t[None, :].expand(H, -1)

    dist = torch.sqrt(
        (lat_grid - SG_LAT_C)**2 +
        ((lon_grid - SG_LON_C) * torch.cos(torch.deg2rad(lat_grid)))**2
    )
    weight = 1.0 + (SG_WEIGHT - 1.0) * torch.exp(-(dist / SG_SIGMA)**2)
    return weight.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]


# =============================================================================
# LOSS FUNCTIONS  (v7: 3 terms only)
# =============================================================================

def loss_edm(pred, target, sigma, sigma_data, weight_map):
    snr_w = min_snr_weight(sigma, sigma_data).reshape(-1, 1, 1, 1)
    err   = (pred - target) ** 2
    if weight_map is not None:
        return (snr_w * err * weight_map).sum() / (weight_map.sum() * pred.size(0) + 1e-8)
    return (snr_w * err).mean()
    
    



def loss_pcc(pred_p, gt_p, weight_map):
    """
    Weighted spatial PCC on precip channel.
    Lightweight (W_PCC=0.2) -- does not fight EDM, just nudges alignment.
    """
    B     = pred_p.size(0)
    w_sqrt = weight_map.squeeze(1).sqrt()        # [H, W]
    p_w   = (pred_p.squeeze(1) * w_sqrt).view(B, -1)
    t_w   = (gt_p.squeeze(1)   * w_sqrt).view(B, -1)
    vx    = p_w - p_w.mean(dim=1, keepdim=True)
    vy    = t_w - t_w.mean(dim=1, keepdim=True)
    corr  = (vx * vy).sum(1) / (
        (vx**2).sum(1).sqrt() * (vy**2).sum(1).sqrt() + 1e-8
    )
    return (1.0 - corr).mean().clamp(0.0, 2.0)


def loss_dry_penalty(pred_p, gt_p, wet_thresh):
    """
    Penalise predicted rain where GT is dry.
    This is the correct fix for ghost rain -- not a loss-weight war.
    Source of ghost rain: embedding noise is too strong -> diffusion fills
    with smooth non-zero values.  This penalty surgically removes that.
    """
    dry = (gt_p < wet_thresh).float()
    return (F.relu(pred_p) * dry).pow(2).mean()

def corrdiff_loss_v7_adaptive(F_x, target_Fx, y_noisy, noise, sigma_r,
                              sigma_data, wet_thresh_precip, weight_map, adaptive_weights):
    
    l_edm = loss_edm(F_x, target_Fx, sigma_r.flatten(), sigma_data, weight_map)
    
    x0_gt = y_noisy - sigma_r * noise
    sigma_flat = sigma_r.flatten()
    c_skip_v = sigma_data**2 / (sigma_flat**2 + sigma_data**2)
    c_out_v = sigma_flat * sigma_data / (sigma_flat**2 + sigma_data**2).sqrt()
    c_skip_v = c_skip_v.reshape(-1,1,1,1)
    c_out_v = c_out_v.reshape(-1,1,1,1)
    
    x0_pred = c_skip_v * y_noisy + c_out_v * F_x
    pred_p = x0_pred[:, PRECIP_CH:PRECIP_CH+1]
    gt_p = x0_gt[:, PRECIP_CH:PRECIP_CH+1]
    
    l_pcc = loss_pcc(pred_p, gt_p, weight_map)
    l_dry = loss_dry_penalty(pred_p, gt_p, wet_thresh_precip)
    
    w = adaptive_weights.get_weights()
    
    total = (w['edm'] * l_edm +
             w['pcc'] * l_pcc +
             w['dry'] * l_dry)
    
    if torch.isnan(total) or torch.isinf(total):
        total = torch.tensor(1.0, device=F_x.device)
    
    metrics = {
        'edm': l_edm.item(),
        'pcc': l_pcc.item(),
        'dry': l_dry.item()
    }
    
    return total, metrics, w   # return weights for logging

# =============================================================================
# ADAPTIVE LOSS WEIGHTS (Same as previous successful version)
# =============================================================================
class AdaptiveLossWeights:
    def __init__(self, init_weights=None, momentum=0.88, min_weight=0.05, max_weight=15.0):
        self.momentum = momentum
        self.min_w = min_weight
        self.max_w = max_weight
        
        self.weights = {
            'edm': init_weights.get('edm', 1.0),
            'pcc': init_weights.get('pcc', 0.15),
            'dry': init_weights.get('dry', 0.20),
        }
        
        self.loss_ma = {k: 1.0 for k in self.weights}   # moving average

    def update(self, loss_dict):
        """Update weights based on current loss values"""
        for k in self.loss_ma:
            if k in loss_dict and loss_dict[k] > 0:
                self.loss_ma[k] = (1 - self.momentum) * loss_dict[k] + \
                                  self.momentum * self.loss_ma[k]
        
        # Normalize
        norm = {k: loss_dict.get(k, 1.0) / (self.loss_ma[k] + 1e-8) for k in self.weights}
        total_norm = sum(norm.values()) + 1e-8
        
        adaptive = {k: (norm[k] / total_norm) * 8.0 for k in self.weights}  # scale to ~8 total
        
        # Smooth blend + safeguards
        for k in self.weights:
            new_w = 0.65 * adaptive[k] + 0.35 * self.weights[k]
            
            if k == 'pcc':
                new_w = max(new_w, 0.08)   # PCC should never become too weak
            elif k == 'dry':
                new_w = max(new_w, 0.10)
            
            self.weights[k] = max(self.min_w, min(self.max_w, new_w))
        
        # Keep EDM relatively stable
        self.weights['edm'] = max(self.weights['edm'], 0.6)

    def get_weights(self):
        return self.weights.copy()

# =============================================================================
# VALIDATION METRICS  (unchanged)
# =============================================================================

def wet_pcc_batch(pred_p, target_p, wet_threshold):
    p_flat = pred_p.reshape(pred_p.size(0), -1)
    t_flat = target_p.reshape(target_p.size(0), -1)
    wet    = t_flat.max(dim=1).values > wet_threshold
    if wet.sum() == 0:
        return torch.tensor(0.0, device=pred_p.device), 0
    p_w, t_w = p_flat[wet], t_flat[wet]
    vx   = p_w - p_w.mean(dim=1, keepdim=True)
    vy   = t_w - t_w.mean(dim=1, keepdim=True)
    corr = (vx * vy).sum(1) / ((vx**2).sum(1).sqrt() * (vy**2).sum(1).sqrt() + 1e-8)
    return corr.mean(), int(wet.sum())


def singapore_pcc(pred_p, gt_p, weight_map, thresh=1.5):
    mask = (weight_map.squeeze() > thresh)
    B    = pred_p.size(0)
    p_sg = pred_p[:, 0][:, mask].reshape(B, -1)
    g_sg = gt_p[:,   0][:, mask].reshape(B, -1)
    vx   = p_sg - p_sg.mean(1, keepdim=True)
    vy   = g_sg - g_sg.mean(1, keepdim=True)
    corr = (vx * vy).sum(1) / ((vx**2).sum(1).sqrt() * (vy**2).sum(1).sqrt() + 1e-8)
    return corr.mean().item()


def intensity_mae(pred_p, target_p, wet_thresh, weight_map=None):
    if weight_map is not None:
        sg_mask = (weight_map.squeeze() > 1.5)
        p = pred_p[:, sg_mask].reshape(pred_p.size(0), -1)
        t = target_p[:, sg_mask].reshape(target_p.size(0), -1)
    else:
        p = pred_p.reshape(pred_p.size(0), -1)
        t = target_p.reshape(target_p.size(0), -1)
    p_max = p.max(dim=1).values
    t_max = t.max(dim=1).values
    wet   = t_max > wet_thresh
    if wet.sum() == 0:
        return 0.0
    return torch.abs(p_max[wet] - t_max[wet]).mean().item()


# =============================================================================
# EMA WITH WARMUP  (unchanged)
# =============================================================================

class EMA:
    def __init__(self, model, decay_final=0.9995, warmup_steps=1000):
        self.decay_final  = decay_final
        self.warmup_steps = warmup_steps
        self.step         = 0
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    def _decay(self):
        frac = min(self.step / max(self.warmup_steps, 1), 1.0)
        return 0.99 + frac * (self.decay_final - 0.99)

    def update(self, model):
        self.step += 1
        d = self._decay()
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1 - d) * p.data + d * self.shadow[n]

    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def backup(self, model):
        self.backup_params = {n: p.data.clone()
                              for n, p in model.named_parameters()
                              if p.requires_grad}

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup_params[n])


# =============================================================================
# MAIN
# =============================================================================

def main():
    global _RANK
    rank, world_size, local_rank = setup_ddp()
    _RANK = rank
    device = torch.device(f"cuda:{local_rank}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    if rank == 0:
        open(LOG_FILE, "w").close()

    BATCH_SIZE    = 32
    LEARNING_RATE = 5e-6
    EPOCHS        = 15000
    PATIENCE      = 300
    P_STD         = 0.8
    N_STEPS_FAST  = 25

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = ClimateDataset(LR_FILES, HR_FILES, rank=rank)
    n = len(dataset)
    train_size = int(0.65 * n)
    val_size   = int(0.15 * n)
    test_size  = n - train_size - val_size

    if rank == 0:
        precip_sample = dataset._hr_arrays["precip"].reshape(-1)
        N_SAMPLE = 1_000_000
        if precip_sample.size > N_SAMPLE:
            idx = np.random.choice(precip_sample.size, N_SAMPLE, replace=False)
            precip_sample = precip_sample[idx]
        WET_THRESH_NORM = float(np.quantile(precip_sample, 0.7))
    else:
        WET_THRESH_NORM = 0.0

    wt_tensor = torch.tensor(WET_THRESH_NORM, device=device)
    dist.broadcast(wt_tensor, src=0)
    WET_THRESH_NORM = wt_tensor.item()

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, _ = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    log(f"Dataset: {n} Train {train_size} Val {val_size} Test {test_size}")
    log(f"WET_THRESH_NORM = {WET_THRESH_NORM:.4f}")

    train_sampler = DistributedSampler(train_set, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_set,   num_replicas=world_size,
                                       rank=rank, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              sampler=train_sampler, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              sampler=val_sampler,   num_workers=2,
                              pin_memory=True, persistent_workers=True)

    # ------------------------------------------------------------------
    # Singapore weight map
    # ------------------------------------------------------------------
    _sample_hr = next(iter(train_loader))[1]
    HR_H = _sample_hr.shape[-2]
    HR_W = _sample_hr.shape[-1]
    del _sample_hr

    singapore_wmap = build_singapore_weight_map(
        HR_H, HR_W, dataset.lats, dataset.lons, device
    )
    log(f"Singapore weight map: {HR_H}x{HR_W} peak={singapore_wmap.max().item():.2f}x")


    embedding_fw = None
    if USE_PAPER_EMBEDDING:

        # All ranks create the object
        embedding_fw = EmbeddingFramework(s_freq=0.25, n_noise_steps=8,
                                          scale_factor=4, device=device)

        # Only rank 0 does the heavy work
        if rank == 0:
            log(" Collecting training data for QDM fitting...")
            lr_list, hr_list = [], []
            with torch.no_grad():
                for i, (lr_b, hr_b) in enumerate(train_loader):
                    if i >= 300:
                        break
                    lr_list.append(lr_b.cpu())
                    hr_list.append(hr_b.cpu())
            lr_all = torch.cat(lr_list)
            hr_all = torch.cat(hr_list)

            # Fit QDM first
            embedding_fw.fit_qdm(lr_all, hr_all, n_quantiles=N_QUANTILES)

            # Then find PSD intersection on QDM-corrected data
            s_freq = find_psd_intersection(train_loader, device,
                                           n_batches=80,
                                           qdm_fw=embedding_fw)
            embedding_fw.s_freq = s_freq

            # Sanity check
            sample_lr = lr_all[:100, PRECIP_CH].float().numpy()
            sample_hr = hr_all[:100, PRECIP_CH].float().numpy()
            qdm_out = embedding_fw._qdm_apply_additive(
                sample_lr,
                embedding_fw.qdm_lr[PRECIP_CH],
                embedding_fw.qdm_hr[PRECIP_CH]
            )
            log(f" QDM sanity: LR mean={sample_lr.mean():.4f} "
                f"-> QDM mean={qdm_out.mean():.4f} "
                f"HR mean={sample_hr.mean():.4f}")

            qdm_state = embedding_fw.state_dict()
            log(f" s_freq={s_freq:.4f}, QDM fitted, broadcasting...")
        else:
            # Non-rank-0 processes: placeholder, will be overwritten by broadcast
            qdm_state = None

        # Broadcast state from rank 0 to all other ranks
        if world_size > 1:
            qdm_list = [qdm_state]
            dist.broadcast_object_list(qdm_list, src=0)
            qdm_state = qdm_list[0]

        # All ranks (including rank 0) load from the broadcast state
        embedding_fw.load_state_dict(qdm_state)
        embedding_fw.betas = embedding_fw.betas.to(device)

        assert embedding_fw._fitted, f"Rank {rank}: embedding_fw not fitted!"
        log(f" Embedding ready rank={rank} s_freq={embedding_fw.s_freq:.4f}")
        
        if rank == 0:
            with torch.no_grad():
                test_hr = hr_all[:16].float()
                f_out   = embedding_fw.f(test_hr)
                corr_vals = []
                for i in range(16):
                    a  = test_hr[i, PRECIP_CH].flatten()
                    b  = f_out[i,  PRECIP_CH].flatten()
                    vx = a - a.mean()
                    vy = b - b.mean()
                    c  = (vx*vy).sum() / ((vx**2).sum().sqrt() * (vy**2).sum().sqrt() + 1e-8)
                    corr_vals.append(c.item())
                mean_corr = float(np.mean(corr_vals))
                log(f" f(HR) vs HR precip correlation = {mean_corr:.4f}")
                log(f" (need >0.3 for model to learn; if low, reduce n_noise_steps)")

                # Also check g(LR) vs f(HR) correlation to verify shared embedding
                g_out = embedding_fw.g(lr_all[:16].float(), hr_size=(test_hr.shape[-2], test_hr.shape[-1]))
                fg_corr_vals = []
                for i in range(16):
                    a  = f_out[i,  PRECIP_CH].flatten()
                    b  = g_out[i,  PRECIP_CH].flatten()
                    vx = a - a.mean()
                    vy = b - b.mean()
                    c  = (vx*vy).sum() / ((vx**2).sum().sqrt() * (vy**2).sum().sqrt() + 1e-8)
                    fg_corr_vals.append(c.item())
                mean_fg_corr = float(np.mean(fg_corr_vals))
                log(f" f(HR) vs g(LR) correlation     = {mean_fg_corr:.4f}")
                log(f" (should be 0.5-0.9; confirms shared embedding space)")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    diffusion_net = SingaporeCorrDiffNet(in_channels=8).to(device)
    # ... rest continues unchanged
    model_core = SingaporeCorrDiffModel(
        diffusion_net,
        sigma_data=0.5, sigma_min=0.002, sigma_max=1.0, rho=_RHO
    ).to(device)

    log("Estimating sigma_data ...")
    sigma_data_est = estimate_sigma_data(train_loader, device, n_batches=50)

    sig_tensor = torch.tensor(sigma_data_est, device=device)
    if world_size > 1:
        dist.all_reduce(sig_tensor, op=dist.ReduceOp.SUM)
        sig_tensor /= world_size
    sigma_data_est = sig_tensor.item()
    if sigma_data_est <= 0:
        sigma_data_est = 0.5

    sigma_min = sigma_data_est * 0.01
    sigma_max = sigma_data_est * 1.2
    P_mean    = math.log(sigma_data_est) + 0.3

    model_core.sigma_data = sigma_data_est
    model_core.sigma_min  = sigma_min
    model_core.sigma_max  = sigma_max

    log(f" sigma_data={sigma_data_est:.4f} sigma_min={sigma_min:.4f} sigma_max={sigma_max:.4f}")
 
    # ------------------------------------------------------------------
    # DDP + Optimizer + EMA
    # ------------------------------------------------------------------
    model_core.diffusion_net = DDP(model_core.diffusion_net,
                                   device_ids=[local_rank],
                                   output_device=local_rank,
                                   find_unused_parameters=False)

    diffusion_net_train = model_core.diffusion_net
    ema = EMA(diffusion_net_train, decay_final=0.9995, warmup_steps=1000)

    optimizer = AdamW(diffusion_net_train.parameters(),
                      lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS,
                                  eta_min=LEARNING_RATE * 0.01)
                                  
    # ====================== ADAPTIVE WEIGHTS ======================
    adaptive_w = AdaptiveLossWeights(
        init_weights={'edm': 1.0, 'pcc': 0.12, 'dry': 0.18},
        momentum=0.87
    )
    # ============================================================

    scaler = GradScaler("cuda")
    if rank == 0:
            betas_np = embedding_fw.betas.cpu().numpy()
            log(f" Beta schedule: first={betas_np[0]:.4f} last={betas_np[-1]:.4f} "
                f"cumulative_noise_var={float(1 - np.prod(1 - betas_np)):.4f}")

    BEST_CKPT_PATH   = os.path.join(SAVE_DIR, "corrdiff_singapore_v7_best.pth")
    LATEST_CKPT_PATH = os.path.join(SAVE_DIR, "corrdiff_singapore_v7_latest.pth")

    best_metric      = float("inf")
    patience_counter = 0
    start_epoch      = 0

    # Resume from v6 checkpoint if available (model arch changed: 12->8 channels,
    # so we do NOT load model weights -- only optimizer/scheduler state if shapes match)
    RESUME_CKPT = os.path.join(SAVE_DIR, "corrdiff_singapore_v7_best.pth")
    if os.path.exists(RESUME_CKPT):
        try:
            ckpt = torch.load(RESUME_CKPT, map_location=device)
            diffusion_net_train.module.load_state_dict(ckpt["model"])
            ema.shadow = ckpt["ema"]
            start_epoch = 0
            best_metric = (0.3 * (1.0 - ckpt.get("val_pcc", 0.0)) +
                           0.5 * (1.0 - ckpt.get("val_sg_pcc", 0.0)) +
                           0.2 * (ckpt.get("val_imae", 2.2) / 2.0))  
            log(f"Resumed from epoch {start_epoch}, best composite={best_metric:.4f}")
        except Exception as e:
            log(f"Could not resume checkpoint (likely arch change): {e}")
            log("Starting from scratch.")

    log("")
    log("=" * 70)
    log("CorrDiff Singapore v7 -- distribution mismatch FIXED")
    #log(f"Loss: {W_EDM} EDM  +  {W_PCC} PCC  +  {W_DRY} DRY PENALTY")
    log(f"Loss: adaptive weights (edm/pcc/dry)")
    log(f"in_channels = 8 (no regressor, full-field target)")
    log(f"SG weight map peak = {singapore_wmap.max().item():.1f}x")
    log("=" * 70)
    log("")

    LOSS_KEYS = ['edm', 'pcc', 'dry', 'extreme']

    # =========================================================================
    # EPOCH LOOP
    # =========================================================================
    for epoch in range(start_epoch, EPOCHS):
        train_sampler.set_epoch(epoch)
        model_core.diffusion_net.train()

        train_loss = 0.0
        loss_accum = {k: 0.0 for k in LOSS_KEYS}
        n_batches  = 0
        t0         = time.time()

        for lr_b, hr_b in train_loader:
            lr_b = lr_b.to(device, non_blocking=True)
            hr_b = hr_b.to(device, non_blocking=True)

            p_max = hr_b[:, PRECIP_CH].reshape(hr_b.size(0), -1).max(dim=1).values
            wet   = p_max > WET_THRESH_NORM
            if wet.sum() == 0:
                continue

            lr_b = lr_b[wet]
            hr_b = hr_b[wet]

            log_sigma = torch.randn(lr_b.size(0), device=device) * P_STD + P_mean
            sigma     = log_sigma.exp().clamp(sigma_min, sigma_max)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.bfloat16):
                F_x, target_Fx, y_noisy, noise, sigma_r = model_core.forward_train(
                    hr_b, lr_b, sigma, embedding_fw=embedding_fw, weight_map=singapore_wmap
                )
                
                loss, metrics, curr_weights = corrdiff_loss_v7_adaptive(
                    F_x, target_Fx, y_noisy, noise, sigma_r,
                    sigma_data_est, WET_THRESH_NORM, singapore_wmap, adaptive_w
                )

            if torch.isnan(loss) or torch.isinf(loss):
                log(f"[W] NaN/Inf at epoch {epoch+1}, skipping batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion_net_train.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(diffusion_net_train)

            train_loss += loss.item()
            for k in LOSS_KEYS:
                loss_accum[k] += metrics.get(k, 0.0)
            n_batches += 1
            
            epoch_metrics = {k: loss_accum[k] for k in ['edm', 'pcc', 'dry']}
            adaptive_w.update(metrics)

        scheduler.step()

        if n_batches == 0:
            continue

        train_loss /= n_batches
        for k in LOSS_KEYS:
            loss_accum[k] /= n_batches

        # ---------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------
        if epoch % 3 == 0 or epoch < 10:
            model_core.diffusion_net.eval()
            ema.backup(diffusion_net_train)
            ema.apply_shadow(diffusion_net_train)

            val_pcc    = 0.0
            val_sg_pcc = 0.0
            val_imae   = 0.0
            val_n      = 0

            with torch.no_grad():
                for lr_b, hr_b in val_loader:
                    lr_b = lr_b.to(device)
                    hr_b = hr_b.to(device)

                    p_max = hr_b[:, PRECIP_CH].reshape(hr_b.size(0), -1).max(dim=1).values
                    wet   = p_max > WET_THRESH_NORM
                    if wet.sum() == 0:
                        continue

                    lr_w = lr_b[wet]
                    hr_w = hr_b[wet]

                    pred_val = model_core.sample(
                        lr_w, n_steps=N_STEPS_FAST,
                        embedding_fw=embedding_fw,
                        weight_map=singapore_wmap
                    )

                    pred_p = pred_val[:, PRECIP_CH]
                    gt_p   = hr_w[:, PRECIP_CH]

                    pcc_val, n_wet = wet_pcc_batch(pred_p, gt_p, WET_THRESH_NORM)
                    sg_pcc = singapore_pcc(
                        pred_val[:, PRECIP_CH:PRECIP_CH+1],
                        hr_w[:, PRECIP_CH:PRECIP_CH+1],
                        singapore_wmap
                    )
                    imae = intensity_mae(pred_p, gt_p, WET_THRESH_NORM,
                                        weight_map=singapore_wmap)

                    val_pcc    += pcc_val.item() * n_wet
                    val_sg_pcc += sg_pcc * n_wet
                    val_imae   += imae   * n_wet
                    val_n      += n_wet

            ema.restore(diffusion_net_train)

            if val_n > 0:
                val_pcc    /= val_n
                val_sg_pcc /= val_n
                val_imae   /= val_n

            #composite = (1.0 - val_pcc) * 0.4 + (1.0 - val_sg_pcc) * 0.6
            #composite = 0.3*(1-PCC) + 0.5*(1-SG-PCC) + 0.2*(I-MAE/2.0)
            composite = (0.3 * (1.0 - val_pcc) + 
                         0.5 * (1.0 - val_sg_pcc) + 
                         0.2 * (val_imae / 2.0))
            elapsed   = time.time() - t0

            curr_w = adaptive_w.get_weights()
            log(f"Ep {epoch+1:5d} | "
                f"loss {train_loss:.4f} "
                f"[edm {loss_accum['edm']:.3f}({curr_w['edm']:.2f}) "
                f"pcc {loss_accum['pcc']:.3f}({curr_w['pcc']:.2f}) "
                f"dry {loss_accum['dry']:.3f}({curr_w['dry']:.2f})] | "
                f"val PCC {val_pcc:.4f} SG-PCC {val_sg_pcc:.4f} "
                f"I-MAE {val_imae:.4f} | "
                f"composite {composite:.4f} | "
                f"{elapsed:.1f}s")

            if rank == 0:
                ckpt = {
                    "epoch":       epoch,
                    "model":       diffusion_net_train.module.state_dict()
                                   if hasattr(diffusion_net_train, "module")
                                   else diffusion_net_train.state_dict(),
                    "ema":         ema.shadow,
                    "opt":         optimizer.state_dict(),
                    "sched":       scheduler.state_dict(),
                    "sigma_data":  sigma_data_est,
                    "sigma_min":   sigma_min,
                    "val_imae": val_imae,
                    "sigma_max":   sigma_max,
                    "val_pcc":     val_pcc,
                    "val_sg_pcc":  val_sg_pcc,
                    "embedding":   embedding_fw.state_dict() if embedding_fw else None,
                }
                torch.save(ckpt, LATEST_CKPT_PATH)

                if composite < best_metric:
                    best_metric      = composite
                    patience_counter = 0
                    torch.save(ckpt, BEST_CKPT_PATH)
                    log(f" >> NEW BEST composite={composite:.4f} "
                        f"(PCC={val_pcc:.4f} SG-PCC={val_sg_pcc:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        log(f"Early stop at epoch {epoch+1}: patience exhausted.")
                        break

        else:
            elapsed = time.time() - t0
            if rank == 0 and epoch % 10 == 0:
                log(f"Ep {epoch+1:5d} | loss {train_loss:.4f} "
                    f"[edm {loss_accum['edm']:.3f} "

                    f"{elapsed:.1f}s")

    log("Training complete.")
    cleanup_ddp()


if __name__ == "__main__":
    main()
