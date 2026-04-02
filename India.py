# -*- coding: ascii -*-
# TrainCorrDiff_India.py  v1
# Adapted from TrainCorrDiff_Singapore.py v5/v6
#
# KEY DIFFERENCES FROM SINGAPORE VERSION:
#
# [1] NO regional weight map  -- all pixels treated equally (uniform weight=1)
#     Singapore had a Gaussian peak of 11x at lat/lon 1.3/103.8.
#     India uses the full domain uniformly; the weight_map argument is simply
#     passed as None everywhere (or ones), so no pixel is boosted.
#
# [2] Only 2 input variables: orography (oro) and precipitation (precip)
#     Singapore had 4 channels: huss, mslp, tas, precip.
#     India has N_CH=2: [oro, precip] where PRECIP_CH=1.
#     The UNet in_channels is adjusted accordingly (see in_channels calc below).
#
# [3] LR/HR file paths updated for India data layout.
#
# [4] QDM fitting uses no singapore_mask (passes None).
#
# [5] Validation metrics: global PCC only (no SG-PCC sub-region metric).
#
# Everything else (EDM math, EmbeddingFramework with QDM, AdaptiveLossWeights,
# PSD intersection, EMA with warmup, DDP, AMP, cosine LR) is IDENTICAL to
# the Singapore v6-ADAPTIVE version.
#
# LOSS PHILOSOPHY (inherited from v5):
#   [1] Spatial PCC loss  -- direct spatial correlation target
#   [2] Intensity loss    -- hard-case pixel errors, top-k selection
#   [3] EDM base loss     -- keeps diffusion math stable
#   [4] SSIM              -- lightweight structural regulariser
#   + sparsity, mass conservation, high-freq PSD (all kept from v6)
#
# Adaptive loss weights auto-balance PCC/Intensity/EDM every epoch.

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

from Dataset import UpscaleDataset
from Network import UNet        
from Network import CorrDiffRegressor


# =============================================================================
# PATHS  -- update these for your India data layout
# =============================================================================

# UpscaleDataset takes a single fine-resolution .nc file and a separate oro file.
# It internally downsamples the fine data 4x to produce coarse inputs.
NC_FILE  = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
ORO_FILE = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"

LOG_FILE = "corrdiff_india_v1.log"
SAVE_DIR = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/CorrDiff/india_v1/"

# -----------------------------------------------------------------------------
# Channel layout:
#   UpscaleDataset returns:
#     batch["fine"]   -> [B, 1, H, W]       fine precip
#     batch["coarse"] -> [B, 1, H/4, W/4]   coarse precip
#     batch["topo"]   -> [B, 1, H, W]       orography (static)
#
#   We treat the model input as precip only (1 channel).
#   Orography is passed as a separate conditioning input to the regressor,
#   matching how CorrDiffRegressor(in_channels=2) uses [coarse, clim_bias]
#   and topo as a separate argument.
#
#   For the diffusion UNet the concatenated input is:
#     [residual_noisy(1) | mu(1) | cond_embedding(1)] = 3 channels
# -----------------------------------------------------------------------------
N_CH        = 1          # precipitation only (topo handled separately by regressor)
PRECIP_CH   = 0          # index of precipitation in the N_CH layout

N_QUANTILES = 500
_RHO        = 5.0

USE_PAPER_EMBEDDING = True
SNR_CLIP            = 5.0

# India: NO regional focus -- weight map is uniform (None throughout)
# (Contrast with Singapore: SG_LAT_C=1.3, SG_LON_C=103.8, SG_WEIGHT=11)


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
# EDM UTILITIES  (unchanged from Singapore v5/v6)
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
# Identical to Singapore v6 except fit_qdm receives singapore_mask=None
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
        # Per-channel QDM tables (global only -- no local sub-region for India)
        self.qdm_lr  = [None] * N_CH
        self.qdm_hr  = [None] * N_CH
        self._fitted = False

    def fit_qdm(self, lr_data, hr_data, wet_thresh=0.0, n_quantiles=N_QUANTILES):
        """
        Fit global QDM for each channel.
        India version: no singapore_mask, no local QDM tables.
        For the orography channel (static), we still fit QDM so the
        upsampled LR oro matches HR oro statistics.
        """
        log("  Fitting GLOBAL QDM for India (oro + precip) ...")
        q_levels = np.linspace(0, 1, n_quantiles)

        for ch in range(N_CH):
            lr_ch = lr_data[:, ch].reshape(-1).cpu().numpy()
            hr_ch = hr_data[:, ch].reshape(-1).cpu().numpy()

            # For precip channel apply wet-day filter before fitting
            if ch == PRECIP_CH and wet_thresh > 0:
                precip_max = hr_data[:, PRECIP_CH].reshape(
                    hr_data.size(0), -1).max(1).values
                wet_mask = (precip_max > wet_thresh).cpu().numpy()
                if wet_mask.sum() > 100:
                    lr_ch = lr_data[wet_mask, ch].reshape(-1).cpu().numpy()
                    hr_ch = hr_data[wet_mask, ch].reshape(-1).cpu().numpy()

            self.qdm_lr[ch] = np.quantile(lr_ch, q_levels)
            self.qdm_hr[ch] = np.quantile(hr_ch, q_levels)

        self._fitted = True
        log("  QDM fitting complete.")

    def _interp(self, arr, q_lr, q_hr):
        return np.interp(arr, q_lr, q_hr)

    def _determine_noise_steps(self):
        return max(10, min(int(self.s_freq * 2.0 * self.n_noise_steps),
                           self.n_noise_steps))

    def f(self, hr_field):
        """Training embedding: downsample HR -> upsample -> add noise."""
        B, C, H, W = hr_field.shape
        sf    = self.scale_factor
        hr_ds = F.interpolate(hr_field, size=(H // sf, W // sf),
                              mode="bilinear", align_corners=False)
        hr_us = F.interpolate(hr_ds, size=(H, W),
                              mode="bilinear", align_corners=False)
        return _apply_noise_steps(hr_us, self.betas, self._determine_noise_steps())

    def g(self, lr_field, hr_size, weight_map=None):
        """
        Inference embedding: QDM-correct LR -> upsample -> add noise.
        weight_map is accepted for API compatibility but ignored for India
        (no local QDM blending -- global QDM only).
        """
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
            lr_qdm[:, ch] = torch.tensor(
                global_map, dtype=lr_field.dtype, device=self.device)

        return _apply_noise_steps(lr_qdm, self.betas, self._determine_noise_steps())

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
        self.betas         = _cosine_betas(self.n_noise_steps, self.device)


# =============================================================================
# PSD INTERSECTION  (unchanged from Singapore v5/v6)
# =============================================================================

def find_psd_intersection(train_loader, device, n_batches=80):
    log("  Computing PSD intersection scale s ...")
    psd_hr_acc = None
    psd_lr_acc = None
    count      = 0

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= n_batches:
                break
            hr_p  = batch["fine"][:, 0].to(device)    # [B, H, W]
            lr_p  = batch["coarse"][:, 0].to(device)  # [B, H/4, W/4]
            H, W  = hr_p.shape[-2], hr_p.shape[-1]
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
# NETWORK WRAPPER
# =============================================================================
# in_channels breakdown for India (2-variable version):
#   residual_noisy : N_CH  = 2
#   mu             : N_CH  = 2
#   cond_ch (emb)  : N_CH  = 2
#   Total          : 6
#
# Compare to Singapore (4-variable):
#   residual_noisy : 4
#   mu             : 4
#   cond_ch        : 4
#   Total          : 12
# =============================================================================

_DIFFUSION_IN_CHANNELS = N_CH * 3   # = 3 for India (1 precip channel x 3 inputs)


class IndiaCorrDiffNet(nn.Module):
    def __init__(self, in_channels=_DIFFUSION_IN_CHANNELS):
        super().__init__()
        # UNet is the diffusion denoiser base class in Network.py
        self.net = UNet(in_channels=in_channels, out_channels=N_CH,
                        global_dim=0)   # no global time features needed here

    def forward(self, x, t):
        H_in, W_in = x.shape[-2], x.shape[-1]
        out = self.net(x, t)
        if out.shape[-2:] != (H_in, W_in):
            out = F.interpolate(out, size=(H_in, W_in),
                                mode="bilinear", align_corners=False)
        return out


# =============================================================================
# MODEL
# =============================================================================

class IndiaCorrDiffModel(nn.Module):
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

    def get_mu(self, lr, topo):
        B = lr.size(0)
        global_feat = torch.zeros(B, 2, device=lr.device)
    
        x_reg = torch.cat([lr, lr], dim=1)
        return self.regressor(x_reg, topo, global_features=global_feat)

    def _mu_at_size(self, lr, topo, H, W):
        mu_raw = self.get_mu(lr, topo)
        if mu_raw.shape[-2:] != (H, W):
            mu_raw = F.interpolate(mu_raw, size=(H, W),
                                   mode="bilinear", align_corners=False)
        return mu_raw

    def forward_train(self, y_fine, lr, topo, sigma, embedding_fw=None):
        H, W = y_fine.shape[-2], y_fine.shape[-1]

        # Resize topo to match HR size if needed
        if topo.shape[-2:] != (H, W):
            topo_hr = F.interpolate(topo, size=(H, W),
                                    mode="bilinear", align_corners=False)
        else:
            topo_hr = topo

        mu = self._mu_at_size(lr, topo_hr, H, W)

        if embedding_fw is not None and USE_PAPER_EMBEDDING:
            cond_ch = embedding_fw.f(y_fine)
        else:
            cond_ch = F.interpolate(lr, size=(H, W),
                                    mode="bilinear", align_corners=False)

        # 5% soft conditioning dropout
        if self.training:
            B = y_fine.size(0)
            drop_mask = 0.95 + 0.05 * torch.rand(B, 1, 1, 1, device=y_fine.device)
            cond_ch   = cond_ch * drop_mask

        residual_target = y_fine - mu.detach()
        noise           = torch.randn_like(residual_target)
        residual_noisy  = residual_target + noise * sigma.reshape(-1, 1, 1, 1)

        # Input: [noisy_residual(1) | mu(1) | conditioning_embedding(1)] = 3 ch
        net_input = torch.cat([residual_noisy, mu, cond_ch], dim=1)

        c_skip, c_out, c_in, c_noise = edm_precond(sigma, self.sigma_data)
        F_x           = self.diffusion_net(c_in * net_input, c_noise)
        residual_pred = c_skip * residual_noisy + c_out * F_x
        residual_pred = torch.clamp(residual_pred, -1.5, 1.5)

        pred = (mu + residual_pred).clamp(min=0.0)
        return pred, mu, residual_pred, residual_target, c_skip, c_out

    @torch.no_grad()
    def sample(self, lr, topo, n_steps=20, embedding_fw=None):
        H_hr = lr.shape[-2] * 4
        W_hr = lr.shape[-1] * 4

        topo_hr = F.interpolate(topo, size=(H_hr, W_hr),
                                mode="bilinear", align_corners=False)
        mu = self._mu_at_size(lr, topo_hr, H_hr, W_hr)

        if embedding_fw is not None and USE_PAPER_EMBEDDING:
            cond_ch = embedding_fw.g(lr, hr_size=(H_hr, W_hr), weight_map=None)
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

        return (mu + x.clamp(-1.5, 1.5)).clamp(min=0.0)


# =============================================================================
# SIGMA_DATA ESTIMATION  (unchanged from Singapore v5/v6)
# =============================================================================

def estimate_sigma_data_hybrid(model_core, embedding_fw, train_loader,
                                device, precip_wet_thresh, n_batches=50):
    stds = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= n_batches:
                break
            lr_b   = batch["coarse"].to(device)   # [B, 1, H/4, W/4]
            hr_b   = batch["fine"].to(device)     # [B, 1, H,   W  ]
            topo_b = batch["topo"].to(device)     # [B, 1, H,   W  ]

            p_max = hr_b[:, 0].reshape(hr_b.size(0), -1).max(dim=1).values
            wet   = p_max > precip_wet_thresh
            if wet.sum() == 0:
                continue
            hr_w   = hr_b[wet]
            lr_w   = lr_b[wet]
            topo_w = topo_b[wet]
            H, W   = hr_w.shape[-2], hr_w.shape[-1]
            if USE_PAPER_EMBEDDING and embedding_fw is not None:
                r = hr_w - embedding_fw.f(hr_w)
            else:
                mu_raw = model_core.get_mu(lr_w, topo_w)
                mu     = F.interpolate(mu_raw, size=(H, W),
                                       mode="bilinear", align_corners=False)
                r = hr_w - mu
            stds.append(r[:, PRECIP_CH].std().item())
    if len(stds) == 0:
        return 0.0
    return float(torch.tensor(stds).mean())


# =============================================================================
# LOSS FUNCTIONS  (v5/v6 -- weight_map=None path used throughout for India)
# =============================================================================

def loss_edm(residual_pred, residual_target, sigma, sigma_data, weight_map=None):
    """EDM denoising loss. weight_map=None -> plain mean (India)."""
    snr_w = min_snr_weight(sigma, sigma_data).reshape(-1, 1, 1, 1)
    err   = (residual_pred - residual_target) ** 2

    if weight_map is not None:
        return (snr_w * err * weight_map).sum() / (
            weight_map.sum() * residual_pred.size(0) + 1e-8)
    return (snr_w * err).mean()


def loss_spatial_pcc(pred_p, gt_p, weight_map=None):
    """
    Global PCC loss.
    For India weight_map=None, so this is a pure unweighted PCC across
    the full domain. No sub-region component (Singapore had 50% Singapore-only).
    """
    B = pred_p.size(0)

    if weight_map is not None:
        w_sqrt = weight_map.squeeze(1).sqrt()
        p_w = (pred_p.squeeze(1) * w_sqrt).view(B, -1)
        t_w = (gt_p.squeeze(1)   * w_sqrt).view(B, -1)
    else:
        p_w = pred_p.squeeze(1).view(B, -1)
        t_w = gt_p.squeeze(1).view(B, -1)

    vx   = p_w - p_w.mean(dim=1, keepdim=True)
    vy   = t_w - t_w.mean(dim=1, keepdim=True)
    corr = (vx * vy).sum(1) / (
        (vx**2).sum(1).sqrt() * (vy**2).sum(1).sqrt() + 1e-8)
    loss = (1.0 - corr).mean()

    return torch.clamp(loss, min=0.0, max=2.0)


def loss_highfreq_psd(pred_p, gt_p, freq_thresh=0.25, eps=1e-8):
    """Penalise deficit of high-frequency power (encourages fine detail)."""
    pf = torch.fft.rfft2(pred_p.float().squeeze(1))
    gf = torch.fft.rfft2(gt_p.float().squeeze(1))

    H, W_h = pf.shape[-2], pf.shape[-1]
    W_full = (W_h - 1) * 2
    fy = torch.fft.fftfreq(H, device=pred_p.device).abs()
    fx = torch.fft.rfftfreq(W_full, device=pred_p.device).abs()
    freq    = (fy[:, None]**2 + fx[None, :]**2).sqrt()
    hf_mask = (freq > freq_thresh).float()

    psd_pred = (pf.abs()**2 * hf_mask).mean()
    psd_gt   = (gf.abs()**2 * hf_mask).mean().clamp(min=eps)

    deficit = torch.log1p(psd_gt / (psd_pred + eps))
    return deficit


def loss_intensity(pred_p, gt_p, weight_map=None, wet_thresh=0.0, k_frac=0.05):
    """
    Intensity loss on hard cases (largest prediction errors).
    India version: no sub-region component -- global top-k errors only.
    weight_map accepted for API compat but ignored.
    """
    B, _, H, W = pred_p.shape
    k_global   = max(1, int(k_frac * H * W))
    wet_mask   = (gt_p > wet_thresh).float()

    gt_flat   = (gt_p * wet_mask).view(B, -1)
    pred_flat = pred_p.view(B, -1)
    error_abs = torch.abs(pred_flat - gt_flat)

    topk_idx    = error_abs.topk(k_global, dim=1).indices
    topk_pred   = pred_flat.gather(1, topk_idx)
    topk_gt     = gt_flat.gather(1, topk_idx)
    loss_global = F.mse_loss(topk_pred, topk_gt)

    return loss_global


def loss_ssim(pred_p, gt_p, window_size=11):
    """
    SSIM loss clamped to [0,1] before computing loss to prevent negative values.
    """
    B, C, H, W = pred_p.shape
    p = pred_p.float().view(B * C, 1, H, W)
    g = gt_p.float().view(B * C, 1, H, W)

    L  = g.max() - g.min() + 1e-8
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

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

    ssim_map = torch.clamp(ssim_map, min=0.0, max=1.0)
    return 1.0 - ssim_map.mean()


# =============================================================================
# ADAPTIVE LOSS WEIGHTS  (identical to Singapore v6)
# =============================================================================

class AdaptiveLossWeights:
    def __init__(self, init_weights=None, momentum=0.8, min_weight=0.1,
                 max_weight=12.0):
        self.momentum = momentum
        self.min_w    = min_weight
        self.max_w    = max_weight

        self.weights = {
            'edm':       init_weights.get('edm',       0.65),
            'pcc':       init_weights.get('pcc',       6.0),
            'intensity': init_weights.get('intensity', 3.0),
            'ssim':      init_weights.get('ssim',      0.2),
        }
        self.loss_ma = {k: 1.0 for k in self.weights}

    def update(self, loss_dict):
        for k in self.loss_ma:
            if k in loss_dict and loss_dict[k] > 0:
                self.loss_ma[k] = (1 - self.momentum) * loss_dict[k] + \
                                  self.momentum * self.loss_ma[k]

        norm       = {k: loss_dict.get(k, 1.0) / (self.loss_ma[k] + 1e-8)
                      for k in self.weights}
        total_norm = sum(norm.values()) + 1e-8
        adaptive   = {k: (norm[k] / total_norm) * 10.0 for k in self.weights}

        for k in self.weights:
            new_w = 0.7 * adaptive[k] + 0.3 * self.weights[k]
            if k == 'pcc':
                new_w = max(new_w, 3.0)
            elif k == 'intensity':
                new_w = max(new_w, 1.5)
            elif k == 'ssim':
                new_w = min(new_w, 0.8)
            self.weights[k] = max(self.min_w, min(self.max_w, new_w))

        self.weights['edm'] = max(self.weights['edm'], 0.4)

    def get_weights(self):
        return self.weights.copy()


# =============================================================================
# COMBINED LOSS (India, uniform weight_map=None)
# =============================================================================

LOSS_KEYS = ['edm', 'pcc', 'intensity', 'ssim', 'sparsity', 'mass', 'hf_psd']


def corrdiff_loss_india(pred_full, y_fine_full,
                        residual_pred, residual_target,
                        sigma, sigma_data,
                        wet_thresh_precip,
                        mu, adaptive_weights):
    """
    India CorrDiff loss.
    Differences from Singapore version:
      - weight_map is None throughout (no regional boosting)
      - No Singapore-only PCC sub-term in loss_spatial_pcc
      - No Singapore-only intensity sub-term in loss_intensity
      - All other terms (edm, ssim, sparsity, mass, hf_psd) identical
    """
    pred_p = pred_full[:, PRECIP_CH:PRECIP_CH+1]
    gt_p   = y_fine_full[:, PRECIP_CH:PRECIP_CH+1]
    r_pred = residual_pred[:, PRECIP_CH:PRECIP_CH+1]
    r_gt   = residual_target[:, PRECIP_CH:PRECIP_CH+1]
    mu_p   = mu[:, PRECIP_CH:PRECIP_CH+1]

    # Core losses (no weight_map -> uniform treatment of all India pixels)
    l_edm  = loss_edm(r_pred, r_gt, sigma, sigma_data, weight_map=None)
    l_pcc  = loss_spatial_pcc(pred_p, gt_p, weight_map=None)
    l_int  = loss_intensity(pred_p, gt_p, weight_map=None,
                            wet_thresh=wet_thresh_precip)
    l_ssim = loss_ssim(pred_p, gt_p)

    # Sparsity: penalise non-zero residuals on dry pixels
    true_dry     = (gt_p < wet_thresh_precip).float()
    ghost_bridge = true_dry * (mu_p > wet_thresh_precip).float()
    l_sparsity   = (F.relu(r_pred) * true_dry).pow(2).mean() + \
                   2.0 * (F.relu(r_pred) * ghost_bridge).pow(2).mean()
    l_sparsity   = torch.clamp(l_sparsity, min=0.0)

    # Mass conservation
    mass_pred = pred_p.sum(dim=[1, 2, 3])
    mass_gt   = gt_p.sum(dim=[1, 2, 3])
    l_mass    = (torch.abs(mass_pred - mass_gt) /
                 (torch.abs(mass_gt) + 1e-6)).mean()
    l_mass    = torch.clamp(l_mass, min=0.0)

    # High-frequency PSD
    l_hf_psd = loss_highfreq_psd(pred_p, gt_p, freq_thresh=0.25)
    l_hf_psd = torch.clamp(l_hf_psd, min=0.0)

    w = adaptive_weights.get_weights()

    total = (w['edm']       * l_edm       +
             w['pcc']       * l_pcc       +
             w['intensity'] * l_int       +
             w['ssim']      * l_ssim      +
             1.2            * l_sparsity  +
             0.25           * l_mass      +
             0.4            * l_hf_psd)

    if total.item() < 0 or torch.isnan(total):
        log(f"[W] Unstable loss: {total.item():.4f}, using safe fallback")
        total = torch.abs(total) + 1.0

    metrics = {
        'edm':       l_edm.item(),
        'pcc':       l_pcc.item(),
        'intensity': l_int.item(),
        'ssim':      l_ssim.item(),
        'sparsity':  l_sparsity.item(),
        'mass':      l_mass.item(),
        'hf_psd':    l_hf_psd.item(),
    }
    return total, metrics, w


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def wet_pcc_batch(pred_p, target_p, wet_threshold):
    """Global PCC on wet-day samples."""
    p_flat = pred_p.reshape(pred_p.size(0), -1)
    t_flat = target_p.reshape(target_p.size(0), -1)
    wet    = t_flat.max(dim=1).values > wet_threshold
    if wet.sum() == 0:
        return torch.tensor(0.0, device=pred_p.device), 0
    p_w, t_w = p_flat[wet], t_flat[wet]
    vx   = p_w - p_w.mean(dim=1, keepdim=True)
    vy   = t_w - t_w.mean(dim=1, keepdim=True)
    corr = (vx * vy).sum(1) / (
        (vx**2).sum(1).sqrt() * (vy**2).sum(1).sqrt() + 1e-8)
    return corr.mean(), int(wet.sum())


def intensity_mae(pred_p, target_p, wet_thresh):
    """
    MAE of peak intensity across the domain.
    India: global domain (no sub-region masking).
    """
    p = pred_p.reshape(pred_p.size(0), -1)
    t = target_p.reshape(target_p.size(0), -1)
    p_max = p.max(dim=1).values
    t_max = t.max(dim=1).values
    wet   = t_max > wet_thresh
    if wet.sum() == 0:
        return 0.0
    return torch.abs(p_max[wet] - t_max[wet]).mean().item()


# =============================================================================
# EMA WITH WARMUP  (unchanged from Singapore v5/v6)
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
    LEARNING_RATE = 2e-5
    EPOCHS        = 15000
    PATIENCE      = 300
    P_STD         = 0.8
    N_STEPS_FAST  = 50

    # ------------------------------------------------------------------
    # Dataset  (UpscaleDataset: single nc_file + oro_file)
    # ------------------------------------------------------------------
    dataset = UpscaleDataset(
        nc_file=NC_FILE,
        oro_file=ORO_FILE,
        downscale_factor=4,
        normalize=True,
        device="cpu",       # precompute on CPU to avoid OOM on large datasets
        split=None,         # we do our own split below
    )
    n          = len(dataset)
    train_size = int(0.65 * n)
    val_size   = int(0.15 * n)
    test_size  = n - train_size - val_size

    # Wet-day threshold: 70th percentile of fine precip pixels (rank-0 computes)
    if rank == 0:
        # Sample from the precomputed fine tensor
        precip_flat = dataset.fine[:, 0].reshape(-1).numpy()
        N_SAMPLE = 1_000_000
        if precip_flat.size > N_SAMPLE:
            idx = np.random.choice(precip_flat.size, N_SAMPLE, replace=False)
            precip_flat = precip_flat[idx]
        WET_THRESH_NORM = float(np.quantile(precip_flat, 0.7))
    else:
        WET_THRESH_NORM = 0.0

    wt_tensor = torch.tensor(WET_THRESH_NORM, device=device)
    dist.broadcast(wt_tensor, src=0)
    WET_THRESH_NORM = wt_tensor.item()

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, _ = random_split(
        dataset, [train_size, val_size, test_size], generator=generator)

    log(f"Dataset: {n}  Train {train_size}  Val {val_size}  Test {test_size}")
    log(f"WET_THRESH_NORM = {WET_THRESH_NORM:.4f}")
    log(f"N_CH={N_CH}  PRECIP_CH={PRECIP_CH}")
    log(f"Diffusion UNet in_channels = {_DIFFUSION_IN_CHANNELS}")

    train_sampler = DistributedSampler(train_set, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_set,   num_replicas=world_size,
                                       rank=rank, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              sampler=train_sampler, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE,
                              sampler=val_sampler, num_workers=2,
                              pin_memory=True, persistent_workers=True)

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
            log("  Collecting training data for QDM fitting...")
            lr_list, hr_list = [], []
            with torch.no_grad():
                for i, batch in enumerate(train_loader):
                    if i >= 300:
                        break
                    lr_list.append(batch["coarse"])
                    hr_list.append(batch["fine"])
            lr_all = torch.cat(lr_list)
            hr_all = torch.cat(hr_list)
            # India: no singapore_mask
            embedding_fw.fit_qdm(
                lr_all, hr_all,
                wet_thresh=WET_THRESH_NORM,
                n_quantiles=N_QUANTILES,
            )
            qdm_state = embedding_fw.state_dict()
        else:
            qdm_state = None

        if world_size > 1:
            qdm_list = [qdm_state]
            dist.broadcast_object_list(qdm_list, src=0)
            qdm_state = qdm_list[0]

        if rank != 0:
            embedding_fw.load_state_dict(qdm_state)

        log(f"  Embedding ready: s_freq={s_freq:.4f}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # CorrDiffRegressor(in_channels=2): expects [coarse, clim_bias] + topo
    regressor = CorrDiffRegressor(
        in_channels=2, out_channels=1,
        base_channels=64, channel_mult=[1, 2, 4],
        num_blocks=2, global_dim=2, dropout=0.0,
    ).to(device)
    reg_ckpt = torch.load("/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/CorrDiff/checkpoints/regressor/regressor_v9.pth", map_location=device)
    # Handle both raw state_dict and checkpoint-wrapped formats
    reg_state = reg_ckpt.get("model_state_dict", reg_ckpt)
    regressor.load_state_dict(reg_state)
    regressor.eval()

    diffusion_net = IndiaCorrDiffNet(in_channels=_DIFFUSION_IN_CHANNELS).to(device)

    model_core = IndiaCorrDiffModel(
        diffusion_net, regressor,
        sigma_data=0.5, sigma_min=0.002, sigma_max=1.0, rho=_RHO
    ).to(device)
    log("Estimating sigma_data ...")
    residual_sigma_data = estimate_sigma_data_hybrid(
        model_core, embedding_fw, train_loader, device,
        WET_THRESH_NORM, n_batches=50)

    has_data   = torch.tensor(1.0 if residual_sigma_data > 0.0 else 0.0, device=device)
    sig_tensor = torch.tensor(residual_sigma_data, device=device)
    if world_size > 1:
        dist.all_reduce(sig_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(has_data,   op=dist.ReduceOp.SUM)
    residual_sigma_data = (sig_tensor / has_data.clamp(min=1.0)).item()

    sigma_min = residual_sigma_data * 0.008
    sigma_max = residual_sigma_data * 1.5
    P_mean    = math.log(residual_sigma_data) + 0.3

    model_core.sigma_data = residual_sigma_data
    model_core.sigma_min  = sigma_min
    model_core.sigma_max  = sigma_max

    log(f"  sigma_data={residual_sigma_data:.4f}  sigma_min={sigma_min:.4f}"
        f"  sigma_max={sigma_max:.4f}")

    # ------------------------------------------------------------------
    # DDP + Optimizer + EMA
    # ------------------------------------------------------------------
    model_core.diffusion_net = DDP(model_core.diffusion_net,
                                   device_ids=[local_rank],
                                   output_device=local_rank,
                                   find_unused_parameters=False)
    diffusion_net_train = model_core.diffusion_net
    ema       = EMA(diffusion_net_train, decay_final=0.9995, warmup_steps=1000)
    optimizer = AdamW(diffusion_net_train.parameters(),
                      lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000,
                                  eta_min=LEARNING_RATE * 0.01)
    scaler    = GradScaler("cuda")

    BEST_CKPT_PATH   = os.path.join(SAVE_DIR, "corrdiff_india_v1_best.pth")
    LATEST_CKPT_PATH = os.path.join(SAVE_DIR, "corrdiff_india_v1_latest.pth")
    RESUME_CKPT      = BEST_CKPT_PATH

    best_metric      = float("inf")
    patience_counter = 0
    start_epoch      = 0

    if os.path.exists(RESUME_CKPT):
        ckpt        = torch.load(RESUME_CKPT, map_location=device)
        diffusion_net_train.module.load_state_dict(ckpt["model"])
        ema.shadow  = ckpt["ema"]
        optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = 1.0 - ckpt.get("val_pcc", 0.0)
        log(f"Resumed from epoch {start_epoch}, best={best_metric:.4f}")

    # Adaptive weights
    adaptive_w = AdaptiveLossWeights(
        init_weights={'edm': 0.62, 'pcc': 6.8, 'intensity': 3.8, 'ssim': 0.18},
        momentum=0.82)

    log("")
    log("=" * 70)
    log("CorrDiff India v1 -- 2 variables (oro + precip), uniform domain")
    log("Adaptive loss weights enabled")
    log(f"UNet in_channels = {_DIFFUSION_IN_CHANNELS}")
    log("NO regional weight map -- full India domain treated uniformly")
    log("=" * 70)
    log("")

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

        for batch in train_loader:
            lr_b   = batch["coarse"].to(device, non_blocking=True)  # [B,1,H/4,W/4]
            hr_b   = batch["fine"].to(device,   non_blocking=True)  # [B,1,H,  W  ]
            topo_b = batch["topo"].to(device,   non_blocking=True)  # [B,1,H,  W  ]

            # Filter to wet-day samples only
            p_max = hr_b[:, PRECIP_CH].reshape(hr_b.size(0), -1).max(dim=1).values
            wet   = p_max > WET_THRESH_NORM
            if wet.sum() == 0:
                continue
            lr_b   = lr_b[wet]
            hr_b   = hr_b[wet]
            topo_b = topo_b[wet]

            log_sigma = torch.randn(lr_b.size(0), device=device) * P_STD + P_mean
            sigma     = log_sigma.exp().clamp(sigma_min, sigma_max)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=torch.bfloat16):
                pred, mu, res_pred, res_target, _, _ = \
                    model_core.forward_train(hr_b, lr_b, topo_b, sigma,
                                             embedding_fw=embedding_fw)

                loss, metrics, curr_weights = corrdiff_loss_india(
                    pred, hr_b, res_pred, res_target,
                    sigma, residual_sigma_data,
                    WET_THRESH_NORM, mu, adaptive_w)

            if torch.isnan(loss) or torch.isinf(loss):
                log(f"[W] NaN/Inf at epoch {epoch+1}, skipping batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                diffusion_net_train.parameters(), 1.0)
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

        # Adapt weights each epoch
        epoch_metrics = {k: loss_accum[k]
                         for k in ['edm', 'pcc', 'intensity', 'ssim']}
        adaptive_w.update(epoch_metrics)

        # ------------------------------------------------------------------
        # VALIDATION
        # ------------------------------------------------------------------
        run_val = (epoch % 5 == 0 or epoch < 10)

        if run_val:
            model_core.diffusion_net.eval()
            ema.backup(diffusion_net_train)
            ema.apply_shadow(diffusion_net_train)

            val_pcc  = 0.0
            val_imae = 0.0
            val_n    = 0

            with torch.no_grad():
                for batch in val_loader:
                    lr_b   = batch["coarse"].to(device)
                    hr_b   = batch["fine"].to(device)
                    topo_b = batch["topo"].to(device)

                    p_max = hr_b[:, PRECIP_CH].reshape(
                        hr_b.size(0), -1).max(dim=1).values
                    wet = p_max > WET_THRESH_NORM
                    if wet.sum() == 0:
                        continue
                    lr_w   = lr_b[wet]
                    hr_w   = hr_b[wet]
                    topo_w = topo_b[wet]

                    pred_val = model_core.sample(
                        lr_w, topo_w, n_steps=N_STEPS_FAST,
                        embedding_fw=embedding_fw)

                    pred_p = pred_val[:, PRECIP_CH]
                    gt_p   = hr_w[:, PRECIP_CH]

                    pcc_val, n_wet = wet_pcc_batch(pred_p, gt_p,
                                                   WET_THRESH_NORM)
                    imae = intensity_mae(pred_p, gt_p, WET_THRESH_NORM)

                    val_pcc  += pcc_val.item() * n_wet
                    val_imae += imae * n_wet
                    val_n    += n_wet

            ema.restore(diffusion_net_train)

            if val_n > 0:
                val_pcc  /= val_n
                val_imae /= val_n

            # Composite metric: lower is better
            composite = 1.0 - val_pcc

            elapsed   = time.time() - t0
            curr_w    = adaptive_w.get_weights()

            log(f"Ep {epoch+1:5d} | "
                f"loss {train_loss:.4f} "
                f"[edm {loss_accum['edm']:.3f}({curr_w['edm']:.2f}) "
                f"pcc {loss_accum['pcc']:.3f}({curr_w['pcc']:.2f}) "
                f"int {loss_accum['intensity']:.3f}({curr_w['intensity']:.2f}) "
                f"ssim {loss_accum['ssim']:.3f}({curr_w['ssim']:.2f}) "
                f"spar {loss_accum['sparsity']:.4f} "
                f"mass {loss_accum['mass']:.4f} "
                f"hfpsd {loss_accum['hf_psd']:.4f}] | "
                f"val PCC {val_pcc:.4f}  I-MAE {val_imae:.4f} | "
                f"composite {composite:.4f} | "
                f"{elapsed:.1f}s")

            if rank == 0:
                ckpt = {
                    "epoch":      epoch,
                    "model":      (diffusion_net_train.module.state_dict()
                                   if hasattr(diffusion_net_train, "module")
                                   else diffusion_net_train.state_dict()),
                    "ema":        ema.shadow,
                    "opt":        optimizer.state_dict(),
                    "sched":      scheduler.state_dict(),
                    "sigma_data": residual_sigma_data,
                    "sigma_min":  sigma_min,
                    "sigma_max":  sigma_max,
                    "val_pcc":    val_pcc,
                    "val_imae":   val_imae,
                    "embedding":  (embedding_fw.state_dict()
                                   if embedding_fw else None),
                    "adaptive_weights": curr_w,
                    "n_ch":       N_CH,
                    "precip_ch":  PRECIP_CH,
                }
                torch.save(ckpt, LATEST_CKPT_PATH)

                if composite < best_metric:
                    best_metric      = composite
                    patience_counter = 0
                    torch.save(ckpt, BEST_CKPT_PATH)
                    log(f"  >> NEW BEST  composite={composite:.4f}"
                        f"  PCC={val_pcc:.4f}  I-MAE={val_imae:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        log(f"Early stop at epoch {epoch+1}: "
                            f"patience exhausted.")
                        break

        else:
            # Non-validation epochs: lightweight log every 10 epochs
            if rank == 0 and epoch % 10 == 0:
                curr_w = adaptive_w.get_weights()
                elapsed = time.time() - t0
                log(f"Ep {epoch+1:5d} | loss {train_loss:.4f} "
                    f"[edm {loss_accum['edm']:.3f}({curr_w['edm']:.2f}) "
                    f"pcc {loss_accum['pcc']:.3f}({curr_w['pcc']:.2f}) "
                    f"int {loss_accum['intensity']:.3f}"
                    f"({curr_w['intensity']:.2f})] | "
                    f"{elapsed:.1f}s")

    log("Training complete.")
    cleanup_ddp()


if __name__ == "__main__":
    main()
