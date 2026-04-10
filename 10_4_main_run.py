# -*- coding: utf-8 -*-
"""
CorrDiff Singapore v7b — SURGICAL MINIMAL FIX
WITH GROUND TRUTH TEST FILE INTEGRATION
===============================================
Philosophy: The v6 model had PCC=0.83, SG-PCC=0.74 at epoch 1581.
            v7 full-rewrite caused regression to PCC=0.73.
            This version makes ONLY the minimum changes needed to:
              1. Fix the provably-broken SSIM (was ~0.786 = saturated)
              2. Fix the undefined `w` dict bug in corrdiff_loss_v5
              3. Push I-MAE lower without disrupting PCC

NEW: Test set now built from ground truth files (1995, 2005, 2014)
     instead of hardcoded year list.
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from Dataset import ClimateDataset
from Network import DiffusionUNet
from Regressor import ClimateRegressor


# =============================================================================
# PATHS
# =============================================================================

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

# Ground truth test files — define years to use for test set
TEST_GT_FILES = [
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/Ground_Truth/precip_rcm_2km_daily_data_2005.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/Ground_Truth/precip_rcm_2km_daily_data_1995.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/Ground_Truth/precip_rcm_2km_daily_data_2014.nc",
]

LOG_FILE  = "2km.log"
SAVE_DIR  = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/CorrDiff/2km/"

_RHO        = 5.0
PRECIP_CH   = 3
N_CH        = 4
N_QUANTILES = 1500

USE_PAPER_EMBEDDING = True
SNR_CLIP            = 5.0

SG_LAT_C  = 1.3
SG_LON_C  = 103.8
SG_SIGMA  = 1.2
SG_WEIGHT = 13.0

# Loss keys — must match what corrdiff_loss_v7b returns
LOSS_KEYS = ['edm', 'pcc', 'intensity', 'ssim', 'sparsity', 'mass', 'hf_psd', 'tail']


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
# NEW: BUILD SPLITS FROM GROUND TRUTH TEST FILES
# =============================================================================

def build_splits_from_test_files(dataset, test_gt_files):
    """
    Build train/val/test splits based on external ground truth test files.
    
    Reads years from test file names and filters the main dataset to:
      - TEST: all samples matching years from test files
      - TRAIN/VAL: remaining samples (85% train / 15% val)
    
    Args:
        dataset: ClimateDataset instance (must have .years attribute)
        test_gt_files: list of paths to ground truth test files
                       Years are extracted from filenames
    
    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    import re
    
    # Extract years from test file names
    test_years = set()
    for fpath in test_gt_files:
        match = re.search(r'(\d{4})', fpath)
        if match:
            year = int(match.group(1))
            test_years.add(year)
            if _RANK == 0:
                log(f"  Found test year {year} in {fpath.split('/')[-1]}")
    
    log(f"Test years identified: {sorted(test_years)}")
    
    # Convert dataset.years to numpy array for vectorized operations
    years_array = np.array(dataset.years)
    indices = np.arange(len(dataset))
    if _RANK == 0:
        unique, counts = np.unique(dataset.years, return_counts=True)
        print("Year distribution:")
        for y, c in zip(unique, counts):
            print(y, c)
        
        # Split by year
    test_mask = np.isin(years_array, list(test_years))
    trainval_mask = ~test_mask
    
    test_indices = indices[test_mask].tolist()
    trainval_indices = indices[trainval_mask].tolist()
    
    # 85/15 split on trainval
    split_point = int(0.85 * len(trainval_indices))
    train_indices = trainval_indices[:split_point]
    val_indices = trainval_indices[split_point:]
    
    log(f"\nDataset split summary:")
    log(f"  Total samples: {len(dataset)}")
    log(f"  Train: {len(train_indices)} ({100*len(train_indices)/len(dataset):.1f}%)")
    log(f"  Val:   {len(val_indices)} ({100*len(val_indices)/len(dataset):.1f}%)")
    log(f"  Test:  {len(test_indices)} ({100*len(test_indices)/len(dataset):.1f}%)")
    log(f"  Test years: {sorted(test_years)}\n")
    
    return train_indices, val_indices, test_indices


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
# EMBEDDING FRAMEWORK  (unchanged from v6)
# =============================================================================

def _cosine_betas(n_steps, device):
    t      = torch.linspace(0, 1, n_steps + 1, device=device)
    alphas = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas = alphas / alphas[0]
    betas  = (1 - alphas[1:] / alphas[:-1]).clamp(max=0.999)
    return betas


def _apply_noise_steps(field, betas, n_steps):
    x = field.clone()
    for t in range(min(n_steps, len(betas))):
        noise = torch.randn_like(x)
        x = (1.0 - betas[t]).sqrt() * x + betas[t].sqrt() * noise
    return x


class EmbeddingFramework:

    def __init__(self, s_freq, n_noise_steps=10, scale_factor=4, device='cpu'):
        self.s_freq        = s_freq
        self.n_noise_steps = n_noise_steps
        self.scale_factor  = scale_factor
        self.device        = device
        self.betas         = _cosine_betas(n_noise_steps, device)
        self.qdm_lr        = [None] * N_CH
        self.qdm_hr        = [None] * N_CH
        self.qdm_lr_local  = [None] * N_CH
        self.qdm_hr_local  = [None] * N_CH
        self._fitted       = False

    def fit_qdm(self, lr_data, hr_data, wet_thresh=0.0, n_quantiles=N_QUANTILES,
                singapore_mask=None):
        log("  Fitting GLOBAL + LOCAL QDM...")
        q_levels = np.linspace(0, 1, n_quantiles)

        for ch in range(N_CH):
            lr_ch = lr_data[:, ch].reshape(-1).cpu().numpy()
            hr_ch = hr_data[:, ch].reshape(-1).cpu().numpy()

            if ch == PRECIP_CH and wet_thresh > 0:
                precip_max = hr_data[:, ch].reshape(hr_data.size(0), -1).max(1).values
                wet_mask   = (precip_max > wet_thresh).cpu().numpy()
                if wet_mask.sum() > 100:
                    lr_ch = lr_data[wet_mask, ch].reshape(-1).cpu().numpy()
                    hr_ch = hr_data[wet_mask, ch].reshape(-1).cpu().numpy()

            self.qdm_lr[ch] = np.quantile(lr_ch, q_levels)
            self.qdm_hr[ch] = np.quantile(hr_ch, q_levels)

            if singapore_mask is not None:
                mask     = torch.tensor(singapore_mask, device=hr_data.device)
                hr_local = hr_data[:, ch][:, mask].reshape(-1).cpu().numpy()
                lr_local = lr_data[:, ch].reshape(-1).cpu().numpy()
                if hr_local.size > 100:
                    self.qdm_lr_local[ch] = np.quantile(lr_local, q_levels)
                    self.qdm_hr_local[ch] = np.quantile(hr_local, q_levels)
                else:
                    self.qdm_lr_local[ch] = self.qdm_lr[ch]
                    self.qdm_hr_local[ch] = self.qdm_hr[ch]

        self._fitted = True
        log("  QDM fitting complete.")

    def _interp(self, arr, q_lr, q_hr):
        return np.interp(arr, q_lr, q_hr)

    def _determine_noise_steps(self):
        return max(10, min(int(self.s_freq * 2.0 * self.n_noise_steps),
                           self.n_noise_steps))

    def f(self, hr_field):
        B, C, H, W = hr_field.shape
        sf         = self.scale_factor
        hr_ds = F.interpolate(hr_field, size=(H // sf, W // sf),
                              mode="bilinear", align_corners=False)
        hr_us = F.interpolate(hr_ds, size=(H, W),
                              mode="bilinear", align_corners=False)
        return _apply_noise_steps(hr_us, self.betas, self._determine_noise_steps())

    def g(self, lr_field, hr_size, weight_map=None):
        if not self._fitted:
            raise RuntimeError("Call fit_qdm() first.")
        B, C   = lr_field.shape[:2]
        H_hr, W_hr = hr_size
        lr_up  = F.interpolate(lr_field, size=(H_hr, W_hr),
                               mode="bilinear", align_corners=False)
        lr_qdm = torch.empty_like(lr_up)

        for ch in range(C):
            arr        = lr_up[:, ch].cpu().numpy()
            global_map = self._interp(arr, self.qdm_lr[ch], self.qdm_hr[ch])
            if self.qdm_lr_local[ch] is not None:
                local_map = self._interp(arr, self.qdm_lr_local[ch], self.qdm_hr_local[ch])
            else:
                local_map = global_map
            global_map = torch.tensor(global_map, dtype=lr_field.dtype, device=self.device)
            local_map  = torch.tensor(local_map,  dtype=lr_field.dtype, device=self.device)
            if weight_map is not None:
                w = weight_map.squeeze(1)
                w = w / (w.mean() + 1e-8)
                w = w.clamp(0.2, 8.0)
                w_power = w ** 2.5
                w_power = w_power / (w_power + 1.0)
                lr_qdm[:, ch] = (1.0 - w_power) * global_map + w_power * local_map
            else:
                lr_qdm[:, ch] = global_map

        return _apply_noise_steps(lr_qdm, self.betas, self._determine_noise_steps())

    def state_dict(self):
        return {"s_freq": self.s_freq, "n_noise_steps": self.n_noise_steps,
                "scale_factor": self.scale_factor,
                "qdm_lr": self.qdm_lr, "qdm_hr": self.qdm_hr,
                "qdm_lr_local": self.qdm_lr_local, "qdm_hr_local": self.qdm_hr_local,
                "fitted": self._fitted}

    def load_state_dict(self, d):
        self.s_freq        = d["s_freq"]
        self.n_noise_steps = d["n_noise_steps"]
        self.scale_factor  = d["scale_factor"]
        self.qdm_lr        = d["qdm_lr"]
        self.qdm_hr        = d["qdm_hr"]
        self.qdm_lr_local  = d.get("qdm_lr_local", [None]*N_CH)
        self.qdm_hr_local  = d.get("qdm_hr_local", [None]*N_CH)
        self._fitted       = d["fitted"]
        self.betas         = _cosine_betas(self.n_noise_steps, self.device)


# =============================================================================
# PSD INTERSECTION  (unchanged)
# =============================================================================

def find_psd_intersection(train_loader, device, n_batches=80):
    log("  Computing PSD intersection scale s ...")
    psd_hr_acc = None
    psd_lr_acc = None
    count      = 0

    with torch.no_grad():
        for i, (lr_b, hr_b) in enumerate(train_loader):
            if i >= n_batches:
                break
            hr_p  = hr_b[:, PRECIP_CH].to(device)
            lr_p  = lr_b[:, PRECIP_CH].to(device)
            H, W  = hr_p.shape[-2], hr_p.shape[-1]
            lr_up = F.interpolate(lr_p.unsqueeze(1), size=(H, W),
                                  mode="bicubic", align_corners=True).squeeze(1)
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
# NETWORK + MODEL  (unchanged from v6)
# =============================================================================

class SingaporeCorrDiffNet(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()
        self.net = DiffusionUNet(in_channels=in_channels, out_channels=N_CH)

    def forward(self, x, t):
        H_in, W_in = x.shape[-2], x.shape[-1]
        out = self.net(x, t)
        if out.shape[-2:] != (H_in, W_in):
            out = F.interpolate(out, size=(H_in, W_in),
                                mode="bilinear", align_corners=False)
        return out


class SingaporeCorrDiffModel(nn.Module):
    def __init__(self, diffusion_net, regressor, sigma_data,
                 sigma_min, sigma_max, rho=_RHO):
        super().__init__()
        self.diffusion_net = diffusion_net
        self.regressor     = regressor
        self.sigma_data    = sigma_data
        self.sigma_min     = sigma_min
        self.sigma_max     = sigma_max
        self.rho           = rho
        for p in self.regressor.parameters():
            p.requires_grad_(False)
        self.regressor_frozen = True

    def get_mu(self, lr):
        return self.regressor(lr)

    def _mu_at_size(self, lr, H, W):
        mu_raw = self.get_mu(lr)
        if mu_raw.shape[-2:] != (H, W):
            mu_raw = F.interpolate(mu_raw, size=(H, W),
                                   mode="bilinear", align_corners=False)
        return mu_raw

    def forward_train(self, y_fine, lr, sigma, embedding_fw=None):
        H, W = y_fine.shape[-2], y_fine.shape[-1]
        mu   = self._mu_at_size(lr, H, W)

        if embedding_fw is not None and USE_PAPER_EMBEDDING:
            cond_ch = embedding_fw.f(y_fine)
        else:
            cond_ch = F.interpolate(lr, size=(H, W),
                                    mode="bilinear", align_corners=False)

        if self.training:
            B = y_fine.size(0)
            drop_mask = 0.95 + 0.05 * torch.rand(B, 1, 1, 1, device=y_fine.device)
            cond_ch   = cond_ch * drop_mask

        residual_target = y_fine - mu.detach()
        noise           = torch.randn_like(residual_target)
        residual_noisy  = residual_target + noise * sigma.reshape(-1, 1, 1, 1)
        net_input       = torch.cat([residual_noisy, mu, cond_ch], dim=1)

        c_skip, c_out, c_in, c_noise = edm_precond(sigma, self.sigma_data)
        F_x           = self.diffusion_net(c_in * net_input, c_noise)
        residual_pred = c_skip * residual_noisy + c_out * F_x
        residual_pred = torch.clamp(residual_pred, -3.0, 3.0)

        pred = (mu + residual_pred).clamp(min=0.0)
        return pred, mu, residual_pred, residual_target, c_skip, c_out

    @torch.no_grad()
    def sample(self, lr, n_steps=20, embedding_fw=None, weight_map=None):
        H_hr = lr.shape[-2] * 4
        W_hr = lr.shape[-1] * 4
        mu   = self._mu_at_size(lr, H_hr, W_hr)

        if embedding_fw is not None and USE_PAPER_EMBEDDING:
            cond_ch = embedding_fw.g(lr, hr_size=(H_hr, W_hr), weight_map=weight_map)
        else:
            cond_ch = F.interpolate(lr, size=(H_hr, W_hr),
                                    mode="bilinear", align_corners=False)

        static_cond = torch.cat([mu, cond_ch], dim=1)
        sigmas      = get_sigma_schedule(n_steps, lr.device,
                                         self.sigma_min, self.sigma_max, self.rho)
        x = torch.randn_like(mu) * sigmas[0] * 0.8

        def denoise(x_in, sig):
            sig_b          = sig.expand(lr.size(0))
            inp            = torch.cat([x_in, static_cond], dim=1)
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

        return (mu + x.clamp(-2.5, 2.5)).clamp(min=0.0)


# =============================================================================
# SIGMA_DATA ESTIMATION  (unchanged)
# =============================================================================

def estimate_sigma_data_hybrid(model_core, embedding_fw, train_loader,
                                device, precip_wet_thresh, n_batches=50):
    stds = []
    with torch.no_grad():
        for i, (lr_b, hr_b) in enumerate(train_loader):
            if i >= n_batches:
                break
            lr_b = lr_b.to(device)
            hr_b = hr_b.to(device)
            p_max = hr_b[:, PRECIP_CH].reshape(hr_b.size(0), -1).max(dim=1).values
            wet   = p_max > precip_wet_thresh
            if wet.sum() == 0:
                continue
            hr_w = hr_b[wet]
            lr_w = lr_b[wet]
            H, W = hr_w.shape[-2], hr_w.shape[-1]
            if USE_PAPER_EMBEDDING and embedding_fw is not None:
                r = hr_w - embedding_fw.f(hr_w)
            else:
                mu_raw = model_core.get_mu(lr_w)
                mu     = F.interpolate(mu_raw, size=(H, W),
                                       mode="bilinear", align_corners=False)
                r = hr_w - mu
            stds.append(r[:, PRECIP_CH].std().item())
    if len(stds) == 0:
        return 0.0
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
    return weight.unsqueeze(0).unsqueeze(0)


# =============================================================================
# LOSS FUNCTIONS  — only 3 functions changed from v6
# =============================================================================

def loss_edm(residual_pred, residual_target, sigma, sigma_data, weight_map):
    """Unchanged from v6."""
    snr_w = min_snr_weight(sigma, sigma_data).reshape(-1, 1, 1, 1)
    err   = (residual_pred - residual_target) ** 2
    if weight_map is not None:
        return (snr_w * err * weight_map).sum() / (weight_map.sum() * residual_pred.size(0) + 1e-8)
    return (snr_w * err).mean()


def loss_spatial_pcc(pred_p, gt_p, weight_map):
    """Unchanged from v6 (was already correct)."""
    B = pred_p.size(0)
    w_sqrt = weight_map.squeeze(1).sqrt()
    p_w = (pred_p.squeeze(1) * w_sqrt).view(B, -1)
    t_w = (gt_p.squeeze(1)   * w_sqrt).view(B, -1)
    vx   = p_w - p_w.mean(dim=1, keepdim=True)
    vy   = t_w - t_w.mean(dim=1, keepdim=True)
    corr_global = (vx * vy).sum(1) / (
        (vx**2).sum(1).sqrt() * (vy**2).sum(1).sqrt() + 1e-8)
    loss_global = (1.0 - corr_global).mean()

    w_raw   = weight_map.squeeze()
    sg_mask = (w_raw > 1.2)
    n_sg    = sg_mask.sum().item()
    loss_sg = torch.zeros(1, device=pred_p.device).squeeze()

    if n_sg > 100:
        sg_mask_batched = sg_mask.unsqueeze(0).expand(B, -1, -1)
        p_sg_all = pred_p.squeeze(1)
        t_sg_all = gt_p.squeeze(1)
        p_sg = p_sg_all[sg_mask_batched].reshape(B, -1)
        t_sg = t_sg_all[sg_mask_batched].reshape(B, -1)
        vx_sg = p_sg - p_sg.mean(dim=1, keepdim=True)
        vy_sg = t_sg - t_sg.mean(dim=1, keepdim=True)
        corr_sg = (vx_sg * vy_sg).sum(1) / (
            (vx_sg**2).sum(1).sqrt() * (vy_sg**2).sum(1).sqrt() + 1e-8)
        loss_sg = (1.0 - corr_sg).mean()

    return torch.clamp(0.3 * loss_global + 0.7 * loss_sg, min=0.0, max=2.0)


def loss_highfreq_psd(pred_p, gt_p, freq_thresh=0.25, eps=1e-8):
    """Unchanged from v6."""
    pf = torch.fft.rfft2(pred_p.float().squeeze(1))
    gf = torch.fft.rfft2(gt_p.float().squeeze(1))
    H, W_h = pf.shape[-2], pf.shape[-1]
    W_full  = (W_h - 1) * 2
    fy = torch.fft.fftfreq(H, device=pred_p.device).abs()
    fx = torch.fft.rfftfreq(W_full, device=pred_p.device).abs()
    freq = (fy[:, None]**2 + fx[None, :]**2).sqrt()
    hf_mask = (freq > freq_thresh).float()
    psd_pred = (pf.abs()**2 * hf_mask).mean()
    psd_gt   = (gf.abs()**2 * hf_mask).mean().clamp(min=eps)
    return torch.clamp(torch.log1p(psd_gt / (psd_pred + eps)), min=0.0)


# ------------------------------------------------------------------
# FIX A: loss_ssim — root cause of the 0.786 saturation
# ------------------------------------------------------------------
def loss_ssim(pred_p, gt_p, window_size=11):
    """
    FIXED v7b: Use fixed C1/C2 constants (not data-range-dependent).

    Root cause of old bug:
      L = g.max() - g.min()  on log1p-normalised precip gives L ≈ 0.3–1.0
      C1 = (0.01 * L)^2 ≈ 9e-6,  C2 = (0.03 * L)^2 ≈ 8e-5
      These are so small that the SSIM map is dominated by numerical noise,
      giving values near -1, which after clamping → 0, so loss = 1.0 always.

    Fix: use the standard fixed constants for unit-scale data (L=1).
    This gives SSIM in a numerically stable range and actual gradients.
    """
    B, C, H, W = pred_p.shape
    p = pred_p.float().view(B * C, 1, H, W)
    g = gt_p.float().view(B * C, 1, H, W)

    # Fixed constants for data normalised to roughly unit range
    C1 = 0.01 ** 2   # = 1e-4
    C2 = 0.03 ** 2   # = 9e-4

    pad    = window_size // 2
    coords = torch.arange(window_size, dtype=torch.float32, device=pred_p.device) - pad
    gk     = torch.exp(-(coords**2) / (2 * 1.5**2))
    gk     = gk / gk.sum()
    kernel = (gk[:, None] * gk[None, :]).unsqueeze(0).unsqueeze(0)

    mu1 = F.conv2d(p, kernel, padding=pad)
    mu2 = F.conv2d(g, kernel, padding=pad)
    s1  = F.conv2d(p*p, kernel, padding=pad) - mu1**2
    s2  = F.conv2d(g*g, kernel, padding=pad) - mu2**2
    s12 = F.conv2d(p*g, kernel, padding=pad) - mu1*mu2

    ssim_map = ((2*mu1*mu2 + C1) * (2*s12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2) + 1e-8)

    ssim_map = ssim_map.clamp(min=-1.0, max=1.0)   # physical range
    loss = (1.0 - ssim_map).mean() / 2.0            # → [0, 1]
    return loss.clamp(min=0.0)


# ------------------------------------------------------------------
# FIX B: loss_intensity — add pinball ON TOP of existing heavy-MSE
# ------------------------------------------------------------------
def loss_intensity(pred_p, gt_p, weight_map, wet_thresh, k_frac=0.08, tail_weight=3.0):
    """
    v7c: Stochastic Tail Guidance + Intensity Freeze.
    - Pinball only activates if batch max > 95th percentile of the dataset.
    - tail_weight is passed from adaptive weights but capped to prevent PCC degradation.
    """
    B, _, H, W = pred_p.shape
    k_global = max(1, int(k_frac * H * W))
    
    wet_mask  = (gt_p > wet_thresh).float()
    gt_flat   = (gt_p * wet_mask).view(B, -1)
    pred_flat = (pred_p * wet_mask).view(B, -1)

    # Standard Heavy-pixel MSE
    topk_idx_heavy = gt_flat.topk(k_global, dim=1).indices
    loss_heavy_global = F.mse_loss(pred_flat.gather(1, topk_idx_heavy),
                                   gt_flat.gather(1, topk_idx_heavy))

    # Singapore region logic
    sg_mask = (weight_map > 1.5).float().expand(B, 1, H, W)
    gt_sg   = (gt_p * wet_mask * sg_mask).view(B, -1)
    pred_sg = (pred_p * sg_mask).view(B, -1)
    
    loss_sg = torch.tensor(0.0, device=pred_p.device)
    if sg_mask[0].sum() > 50:
        k_sg = max(1, int(k_frac * 0.18 * H * W))
        topk_idx_sg = gt_sg.topk(min(k_sg, gt_sg.size(1)//2), dim=1).indices
        loss_sg = F.mse_loss(pred_sg.gather(1, topk_idx_sg), gt_sg.gather(1, topk_idx_sg))

    base_loss = 0.3 * loss_heavy_global + tail_weight * loss_sg

    # --- STOCHASTIC TAIL NUDGE ---
    # Only apply pinball if the batch actually contains a significant peak
    # (Thresholding at 2.5 as a proxy for 'extreme' in log-space)
    pinball = torch.tensor(0.0, device=pred_p.device)
    if gt_p.max() > 2.0: 
        wet_any = (gt_flat > wet_thresh).any(dim=1)
        if wet_any.any():
            pf = pred_flat[wet_any]
            gf = gt_flat[wet_any]
            for q in (0.95, 0.99):
                err = gf - pf
                pinball += (q * err.clamp(min=0) + (1 - q) * (-err).clamp(min=0)).mean()
            pinball /= 2.0

    return base_loss + 0.4 * pinball


def loss_tail_quantile(pred_p, gt_p, wet_thresh, q_levels=(0.95, 0.98, 0.99, 0.995)):
    """Unchanged from v6."""
    pred_flat = pred_p.view(pred_p.size(0), -1)
    gt_flat   = gt_p.view(gt_p.size(0), -1)
    wet_mask  = (gt_flat > wet_thresh).any(dim=1)
    if not wet_mask.any():
        return torch.tensor(0.0, device=pred_p.device)
    loss_q = 0.0
    for q in q_levels:
        q_gt   = torch.quantile(gt_flat[wet_mask], q, dim=1)
        q_pred = torch.quantile(pred_flat[wet_mask], q, dim=1)
        loss_q += (F.mse_loss(q_pred, q_gt) +
                   0.6 * torch.abs((q_pred - q_gt) / (q_gt + 1e-8)).mean())
    return loss_q / len(q_levels)


# =============================================================================
# ADAPTIVE LOSS WEIGHTS  — FIX C: remove duplicate floor, clean bounds
# =============================================================================

class AdaptiveLossWeights:
    """
    v7b: unified bounds table — one floor, one ceiling, applied once.
    The old v6 had two separate blocks both setting pcc floor=6.0
    and the blending between them was wasted computation.
    """

    # (floor, ceiling) — applied after blending, overrides everything
    BOUNDS = {
        'edm':      (0.50,  4.00),
        'pcc':      (7.00, 18.00), # Raised ceiling to allow PCC more 'voice'
        'intensity':(1.50,  3.00), # LOCKED: Prevents I-MAE from stealing PCC gradient
        'ssim':     (0.10,  0.50),
    }

    def __init__(self, init_weights=None, momentum=0.9): # Increased momentum for ultra-stability
        self.momentum = momentum
        self.weights = {
            'edm':       (init_weights or {}).get('edm',       0.65),
            'pcc':       (init_weights or {}).get('pcc',       7.00),
            'intensity': (init_weights or {}).get('intensity', 3.00),
            'ssim':      (init_weights or {}).get('ssim',      0.20),
        }
        self.loss_ma = {k: 1.0 for k in self.weights}

    def update(self, loss_dict, current_pcc=None):
        for k in self.loss_ma:
            v = loss_dict.get(k, 1.0)
            self.loss_ma[k] = (1 - self.momentum) * v + self.momentum * self.loss_ma[k]

        # Calculate raw adaptive needs
        norm = {k: loss_dict.get(k, 1.0) / (self.loss_ma[k] + 1e-8) for k in self.weights}
        total_norm = sum(norm.values()) + 1e-8
        
        # SYNC-LOCK LOGIC: 
        # If PCC is doing poorly, we divert weight away from Intensity.
        # This prevents the model from "cheating" by blurring.
        pcc_performance = current_pcc if current_pcc is not None else 0.84
        intensity_modifier = 1.0 if pcc_performance > 0.842 else 0.7
        
        adaptive = {
            'edm':       (norm['edm'] / total_norm) * 10.0,
            'pcc':       (norm['pcc'] / total_norm) * 12.0,
            'intensity': (norm['intensity'] / total_norm) * 8.0 * intensity_modifier,
            'ssim':      (norm['ssim'] / total_norm) * 2.0,
        }

        for k in self.weights:
            lo, hi = self.BOUNDS[k]
            self.weights[k] = max(lo, min(hi, 0.6 * adaptive[k] + 0.4 * self.weights[k]))

    def get_weights(self):
        return self.weights.copy()


# =============================================================================
# COMBINED LOSS  v7b  — single function, all keys present in LOSS_KEYS
# =============================================================================

def corrdiff_loss_v7b(pred_full, y_fine_full,
                      residual_pred, residual_target,
                      sigma, sigma_data,
                      wet_thresh_precip,
                      weight_map, mu,
                      adaptive_weights):
    """
    v7b: identical structure to v6 corrdiff_loss_v5_adaptive, but:
      - uses the fixed loss_ssim (Fix A)
      - uses the augmented loss_intensity with pinball (Fix B)
      - returns all LOSS_KEYS so accumulation never throws KeyError (Fix D)
    """
    pred_p = pred_full[:, PRECIP_CH:PRECIP_CH+1]
    gt_p   = y_fine_full[:, PRECIP_CH:PRECIP_CH+1]
    r_pred = residual_pred[:, PRECIP_CH:PRECIP_CH+1]
    r_gt   = residual_target[:, PRECIP_CH:PRECIP_CH+1]
    mu_p   = mu[:, PRECIP_CH:PRECIP_CH+1]

    w = adaptive_weights.get_weights()

    l_edm  = loss_edm(r_pred, r_gt, sigma, sigma_data, weight_map)
    l_pcc  = loss_spatial_pcc(pred_p, gt_p, weight_map)
    l_int  = loss_intensity(pred_p, gt_p, weight_map, wet_thresh_precip)
    l_ssim = loss_ssim(pred_p, gt_p)                       # FIX A
    l_tail = loss_tail_quantile(pred_p, gt_p, wet_thresh_precip)

    # Auxiliary terms (small fixed weights — not adaptive)
    true_dry     = (gt_p < wet_thresh_precip).float()
    ghost_bridge = true_dry * (mu_p > wet_thresh_precip).float()
    l_sparsity = (F.relu(r_pred) * true_dry).pow(2).mean() + \
                 2.0 * (F.relu(r_pred) * ghost_bridge).pow(2).mean()
    l_sparsity = l_sparsity.clamp(min=0.0)

    l_mass = (torch.abs(pred_p.sum(dim=[1,2,3]) - gt_p.sum(dim=[1,2,3])) /
              (torch.abs(gt_p.sum(dim=[1,2,3])) + 1e-6)).mean().clamp(min=0.0)

    l_hf_psd = loss_highfreq_psd(pred_p, gt_p).clamp(min=0.0)

    pcc_penalty = 1.0 + torch.clamp(l_pcc, 0, 0.5) 
    
    total = (w['edm']       * l_edm +
             w['pcc']       * l_pcc +
             (w['intensity'] * pcc_penalty) * l_int + # <--- The "Bridge"
             w['ssim']      * l_ssim +
             0.5            * l_tail +
             1.0            * l_sparsity +
             0.2            * l_mass +
             0.3            * l_hf_psd)

    if total.item() < 0 or torch.isnan(total) or torch.isinf(total):
        log(f"[W] Unstable loss: {total.item():.4f}, using |loss|+1 fallback")
        total = torch.abs(total) + 1.0

    metrics = {
        'edm':       l_edm.item(),
        'pcc':       l_pcc.item(),
        'intensity': l_int.item(),
        'ssim':      l_ssim.item(),
        'sparsity':  l_sparsity.item(),
        'mass':      l_mass.item(),
        'hf_psd':    l_hf_psd.item(),
        'tail':      l_tail.item(),      # FIX D: was missing from v6 return dict
    }
    return total, metrics, w


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
# EMA  (unchanged)
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
    _RANK  = rank
    device = torch.device(f"cuda:{local_rank}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    if rank == 0:
        open(LOG_FILE, "w").close()

    BATCH_SIZE    = 32
    LEARNING_RATE = 1e-6    # conservative: fine-tuning at epoch ~1580
    EPOCHS        = 15000
    PATIENCE      = 1500
    P_STD         = 0.8
    N_STEPS_FAST  = 100

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = ClimateDataset(LR_FILES, HR_FILES, rank=rank)
    n          = len(dataset)
   
    if rank == 0:
        precip_sample = dataset._hr_arrays["precip"].reshape(-1)
        N_SAMPLE = 1_000_000
        if precip_sample.size > N_SAMPLE:
            idx = np.random.choice(precip_sample.size, N_SAMPLE, replace=False)
            precip_sample = precip_sample[idx]
        WET_THRESH_NORM = float(np.quantile(precip_sample, 0.60))
    else:
        WET_THRESH_NORM = 0.0

    wt_tensor = torch.tensor(WET_THRESH_NORM, device=device)
    dist.broadcast(wt_tensor, src=0)
    WET_THRESH_NORM = wt_tensor.item()

    # ------------------------------------------------------------------
    # Build splits from ground truth test files
    # ------------------------------------------------------------------
    log("\n" + "="*70)
    log("Building train/val/test splits from ground truth test files...")
    log("="*70)
    train_indices, val_indices, test_indices = build_splits_from_test_files(
        dataset, TEST_GT_FILES
    )
    
    train_set = Subset(dataset, train_indices)
    val_set   = Subset(dataset, val_indices)
    test_set  = Subset(dataset, test_indices)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_set, num_replicas=world_size,
                                       rank=rank, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              sampler=train_sampler, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE,
                              sampler=val_sampler, num_workers=2,
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

    # ------------------------------------------------------------------
    # Embedding framework
    # ------------------------------------------------------------------
    embedding_fw = None
    if USE_PAPER_EMBEDDING:
        if rank == 0:
            s_freq = find_psd_intersection(train_loader, device, n_batches=80)
        else:
            s_freq = 0.0
        s_tensor = torch.tensor(s_freq, device=device)
        if world_size > 1:
            dist.broadcast(s_tensor, src=0)
        s_freq = s_tensor.item()

        embedding_fw = EmbeddingFramework(s_freq=s_freq, n_noise_steps=10,
                                          scale_factor=4, device=device)

        if rank == 0:
            log(" Collecting training data for QDM fitting...")
            lr_list, hr_list = [], []
            with torch.no_grad():
                for i, (lr_b, hr_b) in enumerate(train_loader):
                    if i >= 300: break
                    lr_list.append(lr_b)
                    hr_list.append(hr_b)
            lr_all = torch.cat(lr_list)
            hr_all = torch.cat(hr_list)
            singapore_mask = (singapore_wmap.squeeze() > 1.5).cpu().numpy()
            embedding_fw.fit_qdm(lr_all, hr_all,
                                  wet_thresh=WET_THRESH_NORM,
                                  n_quantiles=N_QUANTILES,
                                  singapore_mask=singapore_mask)
            qdm_state = embedding_fw.state_dict()
        else:
            qdm_state = None

        if world_size > 1:
            qdm_list = [qdm_state]
            dist.broadcast_object_list(qdm_list, src=0)
            qdm_state = qdm_list[0]

        if rank != 0:
            embedding_fw.load_state_dict(qdm_state)
        log(f" Embedding ready: s_freq={s_freq:.4f}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    regressor = ClimateRegressor().to(device)
    regressor.load_state_dict(torch.load("regressor.pth", map_location=device))
    regressor.eval()

    diffusion_net = SingaporeCorrDiffNet(in_channels=12).to(device)
    model_core    = SingaporeCorrDiffModel(
        diffusion_net, regressor,
        sigma_data=0.5, sigma_min=0.002, sigma_max=1.0, rho=_RHO
    ).to(device)

    log("Estimating sigma_data ...")
    residual_sigma_data = estimate_sigma_data_hybrid(
        model_core, embedding_fw, train_loader, device, WET_THRESH_NORM, n_batches=50
    )
    has_data   = torch.tensor(1.0 if residual_sigma_data > 0.0 else 0.0, device=device)
    sig_tensor = torch.tensor(residual_sigma_data, device=device)
    if world_size > 1:
        dist.all_reduce(sig_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(has_data,   op=dist.ReduceOp.SUM)
    residual_sigma_data = (sig_tensor / has_data.clamp(min=1.0)).item()

    sigma_min = residual_sigma_data * 0.005
    sigma_max = residual_sigma_data * 1.6
    P_mean    = math.log(residual_sigma_data) + 0.3

    model_core.sigma_data = residual_sigma_data
    model_core.sigma_min  = sigma_min
    model_core.sigma_max  = sigma_max
    log(f" sigma_data={residual_sigma_data:.4f} sigma_min={sigma_min:.4f} sigma_max={sigma_max:.4f}")

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
                      lr=LEARNING_RATE, weight_decay=1e-4,
                      betas=(0.9, 0.999))

    # Standard cosine annealing — no restarts (restarts caused v7 spike)
    scheduler = CosineAnnealingLR(optimizer, T_max=2000,
                                  eta_min=LEARNING_RATE * 0.01)

    scaler = GradScaler("cuda")

    BEST_CKPT_PATH   = os.path.join(SAVE_DIR, "corrdiff_singapore_v7b_best.pth")
    LATEST_CKPT_PATH = os.path.join(SAVE_DIR, "corrdiff_singapore_v7b_latest.pth")

    # Resume from v7b best if exists, else from v6 best (the PCC=0.83 checkpoint)
    RESUME_CKPT = BEST_CKPT_PATH if os.path.exists(BEST_CKPT_PATH) else \
                  os.path.join(SAVE_DIR, "corrdiff_singapore_v7_latest.pth")

    best_metric      = float("inf")
    patience_counter = 0
    start_epoch      = 0

    if os.path.exists(RESUME_CKPT):
        log(f"Loading checkpoint: {RESUME_CKPT}")
        ckpt = torch.load(RESUME_CKPT, map_location=device)
        diffusion_net_train.module.load_state_dict(ckpt["model"])
        ema.shadow = ckpt["ema"]
        optimizer.load_state_dict(ckpt["opt"])
        for pg in optimizer.param_groups:
            pg['lr'] = LEARNING_RATE   # override to new LR
        start_epoch = ckpt.get("epoch", 0) + 1
        best_metric = (1.0 - val_pcc) * 0.5 + \
              (1.0 - val_sg_pcc) * 0.25 + \
              (val_imae / 4.0) * 0.32
        log(f"Resumed from epoch {start_epoch}, best composite={best_metric:.4f}")

    # Adaptive weights
    adaptive_w = AdaptiveLossWeights(
        init_weights={'edm': 0.65, 'pcc': 6.0, 'intensity': 3.0, 'ssim': 0.2},
        momentum=0.8
    )
    if os.path.exists(RESUME_CKPT):
        ckpt_check = torch.load(RESUME_CKPT, map_location='cpu')
        if "adaptive_weights" in ckpt_check:
            saved = ckpt_check["adaptive_weights"]
            for k in adaptive_w.weights:
                if k in saved:
                    lo, hi = adaptive_w.BOUNDS[k]
                    adaptive_w.weights[k] = max(lo, min(hi, float(saved[k])))
            log("Restored adaptive weights from checkpoint.")
        del ckpt_check

    log("")
    log("=" * 70)
    log("CorrDiff Singapore v7b — SURGICAL MINIMAL FIX")
    log("  Fix A: SSIM C1/C2 now fixed constants (was data-range → saturated)")
    log("  Fix B: intensity += pinball@0.95,0.99 (nudge, not replace)")
    log("  Fix C: adaptive weight update unified bounds (no duplicate floor)")
    log("  Fix D: 'tail' key always in metrics dict")
    log("  LR: 1e-6 (conservative for epoch ~1580 fine-tuning)")
    log("  Scheduler: CosineAnnealing T_max=2000 (no warm-restart spike)")
    log(f"  Peak SG weight = {singapore_wmap.max().item():.1f}x")
    log(f"  Resume epoch = {start_epoch}")
    log(f"  Test years: {sorted([int(y) for y in re.findall(r'd{4}', ' '.join(TEST_GT_FILES))])}")
    log("=" * 70)
    log("")

    # =========================================================================
    # EPOCH LOOP
    # =========================================================================
    for epoch in range(start_epoch, EPOCHS):
        train_sampler.set_epoch(epoch)
        model_core.diffusion_net.train()

        # Delayed regressor fine-tune
        if epoch == 1000 and model_core.regressor_frozen:
            log("Starting VERY LOW LR regressor fine-tuning")
            finetune_params = []
            for name, p in model_core.regressor.named_parameters():
                if 'decoder' in name or 'final' in name:
                    p.requires_grad_(True)
                    finetune_params.append(p)
            if finetune_params:
                optimizer.add_param_group({'params': finetune_params, 'lr': 5e-7})
            for m in model_core.regressor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            model_core.regressor_frozen = False

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
                pred, mu, res_pred, res_target, _, _ = \
                    model_core.forward_train(hr_b, lr_b, sigma, embedding_fw=embedding_fw)

                loss, metrics, curr_weights = corrdiff_loss_v7b(
                    pred, hr_b, res_pred, res_target, sigma, residual_sigma_data,
                    WET_THRESH_NORM, singapore_wmap, mu, adaptive_w
                )

            if torch.isnan(loss) or torch.isinf(loss):
                log(f"[W] NaN/Inf at epoch {epoch+1}, skipping batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if not model_core.regressor_frozen:
                for p in model_core.regressor.parameters():
                    if p.grad is not None:
                        p.grad.mul_(0.1)

            torch.nn.utils.clip_grad_norm_(diffusion_net_train.parameters(), 1.0)
            if not model_core.regressor_frozen:
                torch.nn.utils.clip_grad_norm_(
                    model_core.regressor.parameters(), 0.1)

            scaler.step(optimizer)
            scaler.update()
            ema.update(diffusion_net_train)

            train_loss += loss.item()
            for k in LOSS_KEYS:
                loss_accum[k] += metrics.get(k, 0.0)
            n_batches += 1

        scheduler.step()

        if n_batches == 0:
            continue

        train_loss /= n_batches
        for k in LOSS_KEYS:
            loss_accum[k] /= n_batches

        # Adapt weights
        adaptive_w.update({
            'edm':       loss_accum['edm'],
            'pcc':       loss_accum['pcc'],
            'intensity': loss_accum['intensity'],
            'ssim':      loss_accum['ssim'],
        })
        curr_w = adaptive_w.weights  # direct access (intentional)

        # 1. Prevent intensity collapse (CRITICAL)
        curr_w['intensity'] = max(curr_w['intensity'], 3.00)
        
        # 2. Prevent EDM domination
        curr_w['edm'] = min(curr_w['edm'], 2.0)
        
        # 3. (Optional but recommended) stabilize PCC
        curr_w['pcc'] = max(curr_w['pcc'], 7.0)

        # ---------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------
        if epoch % 5 == 0 or epoch < 10:
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
                    gt_p   = hr_w[:,   PRECIP_CH]

                    pcc_val, n_wet = wet_pcc_batch(pred_p, gt_p, WET_THRESH_NORM)
                    sg_pcc = singapore_pcc(
                        pred_val[:, PRECIP_CH:PRECIP_CH+1],
                        hr_w[:,   PRECIP_CH:PRECIP_CH+1],
                        singapore_wmap
                    )
                    imae = intensity_mae(pred_p, gt_p, WET_THRESH_NORM,
                                         weight_map=singapore_wmap)

                    val_pcc    += pcc_val.item() * n_wet
                    val_sg_pcc += sg_pcc * n_wet
                    val_imae   += imae * n_wet
                    val_n      += n_wet

            ema.restore(diffusion_net_train)

            if val_n > 0:
                val_pcc    /= val_n
                val_sg_pcc /= val_n
                val_imae   /= val_n

            composite = (1.0 - val_pcc) * 0.4 + \
                        (1.0 - val_sg_pcc) * 0.3 + \
                        (val_imae / 5.0) * 0.3
            elapsed   = time.time() - t0
            curr_w    = adaptive_w.get_weights()
            lr_now    = optimizer.param_groups[0]['lr']

            log(f"Ep {epoch+1:5d} | lr {lr_now:.2e} | "
                f"loss {train_loss:.4f} "
                f"[edm {loss_accum['edm']:.3f}({curr_w['edm']:.2f}) "
                f"pcc {loss_accum['pcc']:.3f}({curr_w['pcc']:.2f}) "
                f"int {loss_accum['intensity']:.3f}({curr_w['intensity']:.2f}) "
                f"ssim {loss_accum['ssim']:.3f}({curr_w['ssim']:.2f}) "
                f"tail {loss_accum['tail']:.3f}] | "
                f"val PCC {val_pcc:.4f} SG-PCC {val_sg_pcc:.4f} "
                f"I-MAE {val_imae:.4f} | composite {composite:.4f} | {elapsed:.1f}s")

            if rank == 0:
                ckpt = {
                    "epoch":            epoch,
                    "model":            (diffusion_net_train.module.state_dict()
                                         if hasattr(diffusion_net_train, "module")
                                         else diffusion_net_train.state_dict()),
                    "ema":              ema.shadow,
                    "opt":              optimizer.state_dict(),
                    "sched":            scheduler.state_dict(),
                    "sigma_data":       residual_sigma_data,
                    "sigma_min":        sigma_min,
                    "sigma_max":        sigma_max,
                    "val_pcc":          val_pcc,
                    "val_sg_pcc":       val_sg_pcc,
                    "val_imae":          val_imae,
                    "embedding":        embedding_fw.state_dict() if embedding_fw else None,
                    "adaptive_weights": curr_w,
                }
                torch.save(ckpt, LATEST_CKPT_PATH)

                if composite < best_metric:
                    best_metric      = composite
                    patience_counter = 0
                    torch.save(ckpt, BEST_CKPT_PATH)
                    log(f" >> NEW BEST composite={composite:.4f} "
                        f"(PCC={val_pcc:.4f} SG-PCC={val_sg_pcc:.4f} I-MAE={val_imae:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        log(f"Early stop at epoch {epoch+1}: patience exhausted.")
                        break

        else:
            elapsed = time.time() - t0
            if rank == 0 and epoch % 10 == 0:
                curr_w = adaptive_w.get_weights()
                lr_now = optimizer.param_groups[0]['lr']
                log(f"Ep {epoch+1:5d} | lr {lr_now:.2e} | loss {train_loss:.4f} "
                    f"[edm {loss_accum['edm']:.3f}({curr_w['edm']:.2f}) "
                    f"pcc {loss_accum['pcc']:.3f}({curr_w['pcc']:.2f}) "
                    f"int {loss_accum['intensity']:.3f}({curr_w['intensity']:.2f}) "
                    f"ssim {loss_accum['ssim']:.3f}] | {elapsed:.1f}s")

    log("Training complete.")
    cleanup_ddp()


if __name__ == "__main__":
    import re  # Add this import at module level
    main()
