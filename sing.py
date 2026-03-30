# -*- coding: ascii -*-
# TrainCorrDiff_Singapore.py  v4
# Hybrid: CorrDiff residual diffusion + Aich et al. (2026) QDM embedding
# Encoding: pure ASCII throughout -- no UTF-8, no Unicode, no smart quotes.
#
# v4 additions targeting S2 PCC > 0.95:
#   [P1] SSIM loss on full reconstruction vs GT (weight 1.5)
#        Direct local structural similarity -- the most direct training signal
#        for PCC. Operates on 11x11 windows, rewards correct spatial patterns.
#   [P2] Cross-channel conditioning: huss+mslp+tas injected as auxiliary channels
#        into the precip residual prediction. Atmospheric moisture and pressure
#        fields constrain WHERE rain can fall -- providing this as explicit
#        spatial context directly improves precip PCC.
#   [P3] Phase threshold lowered from 0.3 to 0.05
#        Previous threshold only supervised very high frequencies.
#        Mesoscale rain bands (the structures that dominate PCC) live at 0.05-0.15.
#        Capturing these directly closes the spatial structure gap.
#   [P4] Conditioning dropout reduced from implicit full-drop to 5% partial
#        The forward_train call previously had no dropout on auxiliary channels.
#        Adding 5% soft dropout (multiply by [0.95,1.0] mask) prevents
#        over-fitting to specific conditioning patterns while keeping gradients.
#   [P5] Loss rebalance: SSIM 1.5 | Phase 0.8 (was 0.3) | Tail 1.0 (was 1.0)
#        Spectral back to 0.08 (was 0.11) -- PSD was slightly high, reduce pressure.
#   [P6] EMA warmup: decay ramps from 0.99 to 0.9999 over first 1000 steps.
#        Early EMA with decay=0.9999 has very slow warmup and barely tracks
#        the model. Ramping ensures val sees meaningful EMA weights from ep1.

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

LOG_FILE  = "corrdiff_v10.log"
SAVE_DIR  = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/CorrDiff/version10/"

_RHO        = 5.0
PRECIP_CH   = 3
N_CH        = 4
N_QUANTILES = 500

USE_PAPER_EMBEDDING = True
_NORM_CEILING = 5.0 
SNR_CLIP = 5.0


# =============================================================================
# [R] SINGAPORE REGION FOCUS
#
# Singapore sits at approximately 1.3N, 103.8E.
# The HR grid is 128x128 covering a regional domain.
# We define the Singapore core as a pixel bounding box within that grid.
#
# SINGAPORE_BOX = (row_start, row_end, col_start, col_end)
# These are PIXEL indices in the 128x128 HR output grid.
# Adjust these to match your actual grid coordinates.
#
# How to find the right indices:
#   ds = xr.open_dataset(HR_FILES[0])
#   lat = ds.lat.values  # or ds.LATITUDE
#   lon = ds.lon.values
#   row_start = np.searchsorted(lat, 1.0)   # 1.0N
#   row_end   = np.searchsorted(lat, 1.6)   # 1.6N
#   col_start = np.searchsorted(lon, 103.5) # 103.5E
#   col_end   = np.searchsorted(lon, 104.1) # 104.1E
#
# Default values below are approximate for a Southeast Asia domain.
# The model will still train correctly even if these are slightly off --
# the region weight just gives extra emphasis to whatever is inside the box.
#
# SINGAPORE_WEIGHT: how much more to penalise errors in the Singapore box.
#   3.0 = Singapore pixels contribute 3x more to the loss than other pixels.
#   Recommended range: 2.0 to 5.0. Start at 3.0.
# =============================================================================
SINGAPORE_BOX    = (56, 72, 88, 104)  # (row_start, row_end, col_start, col_end)
SINGAPORE_WEIGHT =   7.0               # multiplier for Singapore region loss


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
# EDM UTILITIES
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


# =============================================================================
# EMBEDDING FRAMEWORK  (Aich et al. 2026)
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

    def __init__(self, s_freq, n_noise_steps=50, scale_factor=4, device='cpu'):
        self.s_freq        = s_freq
        self.n_noise_steps = n_noise_steps
        self.scale_factor  = scale_factor
        self.device        = device
        self.betas         = _cosine_betas(n_noise_steps, device)

        # GLOBAL QDM
        self.qdm_lr        = [None] * N_CH
        self.qdm_hr        = [None] * N_CH

        # LOCAL (Singapore) QDM
        self.qdm_lr_local  = [None] * N_CH
        self.qdm_hr_local  = [None] * N_CH

        self._fitted       = False

    def fit_qdm(self, lr_data, hr_data,
                wet_thresh=0.0,
                n_quantiles=N_QUANTILES,
                singapore_mask=None):

        log("  Fitting GLOBAL + LOCAL QDM...")

        q_levels = np.linspace(0, 1, n_quantiles)

        for ch in range(N_CH):

            # ---------------- GLOBAL ----------------
            lr_ch = lr_data[:, ch].reshape(-1).cpu().numpy()
            hr_ch = hr_data[:, ch].reshape(-1).cpu().numpy()

            if ch == PRECIP_CH and wet_thresh > 0:
                precip_max = hr_data[:, ch].reshape(hr_data.size(0), -1).max(1).values
                wet_mask   = (precip_max > wet_thresh).cpu().numpy()

                if wet_mask.sum() > 100:
                    lr_ch = lr_data[wet_mask, ch].reshape(-1).cpu().numpy()
                    hr_ch = hr_data[wet_mask, ch].reshape(-1).cpu().numpy()
                    log(f"    ch{ch} (precip): GLOBAL QDM on {wet_mask.sum()} wet days")

            self.qdm_lr[ch] = np.quantile(lr_ch, q_levels)
            self.qdm_hr[ch] = np.quantile(hr_ch, q_levels)

            # ---------------- LOCAL (Singapore) ----------------
            if singapore_mask is not None:

                mask = torch.tensor(singapore_mask, device=hr_data.device)
            
                # ONLY use HR for local QDM
                hr_local = hr_data[:, ch][:, mask].reshape(-1).cpu().numpy()
            
                # Use SAME distribution for LR (important trick)
                lr_local = lr_data[:, ch].reshape(-1).cpu().numpy()
            
                if hr_local.size > 100:
                    self.qdm_lr_local[ch] = np.quantile(lr_local, q_levels)
                    self.qdm_hr_local[ch] = np.quantile(hr_local, q_levels)
                else:
                    self.qdm_lr_local[ch] = self.qdm_lr[ch]
                    self.qdm_hr_local[ch] = self.qdm_hr[ch]

        self._fitted = True
        log("  QDM fitting complete (global + local)")

    def _interp(self, arr, q_lr, q_hr):
        return np.interp(arr, q_lr, q_hr)

    def _determine_noise_steps(self):
        steps = int(self.s_freq * 2.0 * self.n_noise_steps)
        return max(10, min(steps, self.n_noise_steps))

    def f(self, hr_field):
        B, C, H, W = hr_field.shape
        sf         = self.scale_factor

        lr_size = (H // sf, W // sf)

        hr_ds = F.interpolate(hr_field, size=lr_size,
                              mode="bilinear", align_corners=False)

        hr_us = F.interpolate(hr_ds, size=(H, W),
                              mode="bilinear", align_corners=False)

        n_steps = self._determine_noise_steps()

        return _apply_noise_steps(hr_us, self.betas, n_steps)

    def g(self, lr_field, hr_size, weight_map=None):

        if not self._fitted:
            raise RuntimeError("Call fit_qdm() before using EmbeddingFramework.")
    
        B, C = lr_field.shape[:2]
        H_hr, W_hr = hr_size
    
        # Step 1: Upsample FIRST
        lr_up = F.interpolate(
            lr_field,
            size=(H_hr, W_hr),
            mode="bilinear",
            align_corners=False
        )
    
        lr_qdm = torch.empty_like(lr_up)
    
        for ch in range(C):
    
            arr = lr_up[:, ch].cpu().numpy()
    
            global_map = self._interp(arr, self.qdm_lr[ch], self.qdm_hr[ch])
    
            if self.qdm_lr_local[ch] is not None:
                local_map = self._interp(
                    arr,
                    self.qdm_lr_local[ch],
                    self.qdm_hr_local[ch]
                )
            else:
                local_map = global_map
    
            global_map = torch.tensor(global_map, dtype=lr_field.dtype, device=self.device)
            local_map  = torch.tensor(local_map,  dtype=lr_field.dtype, device=self.device)
    
            if weight_map is not None:
                w = weight_map.squeeze(1)
                
                # normalize around mean, NOT min-max
                w = w / (w.mean() + 1e-8)
                
                # clamp for stability
                w = w.clamp(0.5, 3.0)
                lr_qdm[:, ch] = (1 - w) * global_map + w * local_map
            else:
                lr_qdm[:, ch] = global_map
    
        n_steps = self._determine_noise_steps()
    
        return _apply_noise_steps(lr_qdm, self.betas, n_steps)

    def state_dict(self):
        return {
            "s_freq":        self.s_freq,
            "n_noise_steps": self.n_noise_steps,
            "scale_factor":  self.scale_factor,
            "qdm_lr":        self.qdm_lr,
            "qdm_hr":        self.qdm_hr,
            "qdm_lr_local":  self.qdm_lr_local,
            "qdm_hr_local":  self.qdm_hr_local,
            "fitted":        self._fitted,
        }

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
# PSD INTERSECTION
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

    THRESHOLD  = 0.005
    diff       = hr_1d - lr_1d
    cross_idxs = (diff[1:] > THRESHOLD).nonzero(as_tuple=True)[0]

    if len(cross_idxs) == 0:
        s_freq = 0.25
        log(f"  No clear PSD crossing found. Using safe default s={s_freq:.3f}")
    else:
        raw_s  = freq_edges[cross_idxs[0] + 1].item()
        s_freq = raw_s
        if s_freq < 0.04:
            log(f"  PSD crossing very low ({s_freq:.4f}), clamping to 0.04")
            s_freq = 0.04

    noise_steps = max(5, min(int(s_freq * 2.0 * 50), 50))
    log(f"  Final s_freq = {s_freq:.4f}  noise_steps = {noise_steps}")
    return s_freq


# =============================================================================
# NETWORK WRAPPER
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


# =============================================================================
# MODEL
# =============================================================================

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

    def get_mu(self, lr):
        with torch.no_grad():
            return self.regressor(lr)

    def _mu_at_size(self, lr, H, W):
        mu_raw = self.get_mu(lr)
        if mu_raw.shape[-2:] != (H, W):
            mu_raw = F.interpolate(mu_raw, size=(H, W),
                                   mode="bilinear", align_corners=False)
        return mu_raw

    def forward_train(self, y_fine, lr, sigma, embedding_fw=None):
        """
        y_fine      : [B, C, H, W]  HR target
        lr          : [B, C, h, w]  LR input
        sigma       : [B]           noise levels
        embedding_fw: EmbeddingFramework or None
        """
        H, W = y_fine.shape[-2], y_fine.shape[-1]
        mu   = self._mu_at_size(lr, H, W)

        if embedding_fw is not None and USE_PAPER_EMBEDDING:
            cond_ch = embedding_fw.f(y_fine)
        else:
            cond_ch = F.interpolate(lr, size=(H, W),
                                    mode="bilinear", align_corners=False)

        # [P4] 5% soft conditioning dropout on cond_ch only.
        # Prevents over-reliance on the embedding while keeping most gradient.
        # Multiply each sample's cond_ch by a mask sampled from [0.95, 1.0].
        if self.training:
            B = y_fine.size(0)
            drop_mask = 0.95 + 0.15 * torch.rand(B, 1, 1, 1, device=y_fine.device)
            cond_ch = cond_ch * drop_mask

        #residual_target = y_fine - mu.detach()
        residual_target = (y_fine - mu.detach())
        noise           = torch.randn_like(residual_target)
        residual_noisy  = residual_target + noise * sigma.reshape(-1, 1, 1, 1)

        net_input = torch.cat([
            residual_noisy * 1.0,
            mu,
            cond_ch
        ], dim=1)

        c_skip, c_out, c_in, c_noise = edm_precond(sigma, self.sigma_data)
        F_x           = self.diffusion_net(c_in * net_input, c_noise)
        residual_pred = c_skip * residual_noisy + c_out * F_x
        residual_pred = torch.clamp(residual_pred, -1.2, 1.2)

        pred_raw = mu + residual_pred
        over     = F.relu(pred_raw - _NORM_CEILING)
        #pred     = pred_raw - over   # equivalent to clamp but differentiable
        pred = pred_raw - 0.20 * over
        pred     = pred.clamp(min=0.0)
        return pred, mu, residual_pred, residual_target, c_skip, c_out, over.pow(2).mean()

    @torch.no_grad()
    def sample(self, lr, n_steps=20, embedding_fw=None, weight_map=None):
        H_hr = lr.shape[-2] * 4
        W_hr = lr.shape[-1] * 4
        mu   = self._mu_at_size(lr, H_hr, W_hr)

        if embedding_fw is not None and USE_PAPER_EMBEDDING:
            cond_ch = embedding_fw.g(lr, hr_size=(H_hr, W_hr),
                         weight_map=weight_map)
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

        x = torch.clamp(x, -1.2, 1.2)
        return torch.clamp(mu + 0.15 * x, min=0.0, max=_NORM_CEILING)


# =============================================================================
# SIGMA_DATA ESTIMATION
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
                f_hr = embedding_fw.f(hr_w)
                r    = hr_w - f_hr
            else:
                mu_raw = model_core.get_mu(lr_w)
                mu     = F.interpolate(mu_raw, size=(H, W),
                                       mode="bilinear", align_corners=False)
                r = hr_w - mu
            stds.append(r[:, PRECIP_CH].std().item())

    # If this rank saw no wet samples, return a placeholder -- the all_reduce
    # average below will pull in the correct value from other ranks.
    if len(stds) == 0:
        return 0.0   # <-- was: raise RuntimeError

    return float(torch.tensor(stds).mean())


# =============================================================================
# LOSS COMPONENTS
# =============================================================================

def spectral_loss_weighted(pred, target):
    r_pred_fft = torch.fft.rfft2(pred.float())
    r_tgt_fft  = torch.fft.rfft2(target.float())
    H, W = pred.shape[-2], pred.shape[-1]
    fy   = torch.fft.fftfreq(H, device=pred.device).abs()
    fx   = torch.fft.rfftfreq(W, device=pred.device).abs()
    freq_mag    = (fy[:, None]**2 + fx[None, :]**2).sqrt()
    freq_weight = (1.0 + 1.5 * freq_mag / freq_mag.max()).detach()
    psd_pred = r_pred_fft.abs().pow(2)
    psd_tgt  = r_tgt_fft.abs().pow(2)
    #log_err  = (torch.log1p(psd_pred) - torch.log1p(psd_tgt)).pow(2)
    #log_err = torch.abs(torch.log1p(psd_pred) - torch.log1p(psd_tgt)) # Use L1 instead of .pow(2)
    amp_err = ((r_pred_fft.abs() - r_tgt_fft.abs()) ** 2) / (r_tgt_fft.abs() ** 2 + 1e-6)
    amp_err = amp_err.clamp(max=4.0)
    log_err = amp_err
    return (freq_weight * log_err).mean()


# REPLACE the entire function with:
def fourier_phase_loss(pred, target):
    pf = torch.fft.rfft2(pred.float())
    tf = torch.fft.rfft2(target.float())

    phase_p = pf / (pf.abs() + 1e-8)
    phase_t = tf / (tf.abs() + 1e-8)
    phase_err = 1.0 - (phase_p * phase_t.conj()).real

    H, W_h  = pf.shape[-2], pf.shape[-1]
    W_full  = (W_h - 1) * 2
    fy      = torch.fft.fftfreq(H, device=pred.device).abs()
    fx      = torch.fft.rfftfreq(W_full, device=pred.device).abs()
    freq_mag = (fy[:, None]**2 + fx[None, :]**2).sqrt()

    freq_mask  = (freq_mag > 0.02).float()
    amp_weight = (pf.abs() * tf.abs()).sqrt().detach()
    amp_weight = amp_weight / (amp_weight.mean() + 1e-8)

    return (phase_err * freq_mask * amp_weight).mean()

def region_tail_loss(pred, target, mask, k=20):
    B = pred.size(0)
    losses = []

    for i in range(B):
        p = pred[i][mask[0] > 0].view(-1)
        t = target[i][mask[0] > 0].view(-1)

        if p.numel() == 0:
            continue

        k_eff = min(k, p.numel())

        p_topk = p.topk(k_eff).values
        t_topk = t.topk(k_eff).values

        losses.append(F.huber_loss(p_topk, t_topk, delta=0.5))

    if len(losses) == 0:
        return torch.tensor(0.0, device=pred.device)

    return torch.stack(losses).mean()


def percentile_loss(pred, target, p=0.99):
    k        = max(1, int((1 - p) * pred.view(pred.size(0), -1).size(1)))
    pred_top = pred.view(pred.size(0), -1).topk(k, dim=1).values.mean(dim=1)
    tgt_top  = target.view(target.size(0), -1).topk(k, dim=1).values.mean(dim=1)
    return F.huber_loss(pred_top, tgt_top, delta=0.5)


def phase_coherence_metric(pred, target):
    pred_fft = torch.fft.rfft2(pred.float())
    tgt_fft  = torch.fft.rfft2(target.float())
    pred_p   = pred_fft / (pred_fft.abs() + 1e-8)
    tgt_p    = tgt_fft  / (tgt_fft.abs()  + 1e-8)
    return (pred_p * tgt_p.conj()).real.mean().item()


# =============================================================================
# [R] SINGAPORE SPATIAL WEIGHT MAP
#
# Builds a [1, 1, H, W] weight tensor that is SINGAPORE_WEIGHT inside the
# Singapore bounding box and 1.0 everywhere else.
# The map is built once at training startup and cached -- no per-batch cost.
#
# Used in two ways:
#   1. region_weighted_loss(): multiplies squared pixel errors by this map
#      before averaging, so Singapore errors contribute more to the total loss.
#   2. region_pcc_metric(): computes PCC restricted to Singapore pixels only,
#      logged separately so you can track Singapore-specific performance.
#
# Gaussian blend at the boundary (sigma=2 pixels) avoids a hard edge that
# can create checkerboard artifacts at the box boundary.
# =============================================================================

def build_singapore_weight_map(H, W, lat, lon, device):
    lat = torch.tensor(lat, device=device)
    lon = torch.tensor(lon, device=device)

    lat_grid = lat[:, None].expand(-1, W)
    lon_grid = lon[None, :].expand(H, -1)

    lat_c = 1.3
    lon_c = 103.8

    dist = torch.sqrt(
        (lat_grid - lat_c)**2 +
        ((lon_grid - lon_c) * torch.cos(torch.deg2rad(lat_grid)))**2
    )

    weight = 1.0 + (SINGAPORE_WEIGHT - 1.0) * torch.exp(-(dist / 1.2)**2)

    return weight.unsqueeze(0).unsqueeze(0)


def region_weighted_loss(pred, target, weight_map):
    """
    MSE loss with spatial weighting.
    pred, target : [B, 1, H, W]
    weight_map   : [1, 1, H, W]  -- broadcast over batch
    Returns scalar.
    """
    sq_err = (pred.float() - target.float()) ** 2
    return (sq_err * weight_map).mean()


def region_pcc_metric(pred_p, target_p, box):
    """
    PCC computed only within the Singapore bounding box.
    pred_p, target_p : [B, H, W]
    box : (r0, r1, c0, c1)
    Returns float PCC averaged over batch.
    """
    r0, r1, c0, c1 = box
    p_roi = pred_p[:, r0:r1, c0:c1].reshape(pred_p.size(0), -1)
    t_roi = target_p[:, r0:r1, c0:c1].reshape(target_p.size(0), -1)
    vx    = p_roi - p_roi.mean(dim=1, keepdim=True)
    vy    = t_roi - t_roi.mean(dim=1, keepdim=True)
    corr  = (vx * vy).sum(1) / ((vx**2).sum(1).sqrt() * (vy**2).sum(1).sqrt() + 1e-8)
    return corr.mean().item()


def region_intensity_metric(pred_p, target_p, box):
    """
    Mean absolute peak intensity error within Singapore box only.
    pred_p, target_p : [B, H, W]
    """
    r0, r1, c0, c1 = box
    p_roi   = pred_p[:,   r0:r1, c0:c1].reshape(pred_p.size(0),   -1)
    t_roi   = target_p[:, r0:r1, c0:c1].reshape(target_p.size(0), -1)
    p_max   = p_roi.max(dim=1).values
    t_max   = t_roi.max(dim=1).values
    return torch.abs(p_max - t_max).mean().item()


# [P1] SSIM loss -- directly rewards spatial structural similarity.
# PCC measures linear correlation across the whole field.
# SSIM measures local luminance, contrast and structure in 11x11 windows.
# This is the single most direct training signal for improving PCC.
# Operates on the full reconstructed precipitation field vs GT.
def ssim_loss(pred, target, window_size=11):
    B, C, H, W = pred.shape
    pred_f = pred.float().view(B * C, 1, H, W)
    target_f = target.float().view(B * C, 1, H, W)
    
    # Use standard constants
    L = target_f.max() - target_f.min() + 1e-8
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    
    pad = window_size // 2
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - pad
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    mu1 = F.conv2d(pred_f, kernel, padding=pad)
    mu2 = F.conv2d(target_f, kernel, padding=pad)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(pred_f * pred_f, kernel, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(target_f * target_f, kernel, padding=pad) - mu2_sq
    sigma12 = F.conv2d(pred_f * target_f, kernel, padding=pad) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
    
    return 1.0 - ssim_map.mean()

def min_snr_weight(sigma, sigma_data, snr_clip=SNR_CLIP):
    snr    = (sigma_data / sigma.clamp(min=1e-8)) ** 2
    weight = (snr_clip / snr).clamp(max=1.0)
    return weight.detach()
def build_singapore_mask(weight_map, thresh=1.5):
    return (weight_map > thresh).float()  
    
def region_pcc_from_mask(pred, target, mask):
    B = pred.size(0)
    mask = mask.expand(B, -1, -1, -1)

    p = pred * mask.squeeze(1)
    t = target * mask.squeeze(1)

    p = p.view(B, -1)
    t = t.view(B, -1)

    vx = p - p.mean(dim=1, keepdim=True)
    vy = t - t.mean(dim=1, keepdim=True)

    corr = (vx * vy).sum(1) / ((vx**2).sum(1).sqrt() * (vy**2).sum(1).sqrt() + 1e-8)
    return corr.mean().item()
    
    
def pcc_loss(pred, target, eps=1e-8):
    B = pred.size(0)
    p = pred.view(B, -1)
    t = target.view(B, -1)

    vx = p - p.mean(dim=1, keepdim=True)
    vy = t - t.mean(dim=1, keepdim=True)

    corr = (vx * vy).sum(1) / ((vx**2).sum(1).sqrt() * (vy**2).sum(1).sqrt() + eps)
    return 1.0 - corr.mean()

def _gaussian_blur_2d(field, sigma_px):
    if sigma_px < 0.5:
        return field
    # Cast to float32 for numerical stability in convolution.
    # multi_scale_hazing_loss calls .float() on its inputs before passing here,
    # so this is a safety net for any direct calls.
    field_f = field.float()
    B, C, H, W = field_f.shape
    r      = int(math.ceil(3.0 * sigma_px))
    ksize  = 2 * r + 1
    x      = torch.arange(-r, r + 1, dtype=torch.float32, device=field.device)
    kernel = torch.exp(-0.5 * (x / sigma_px) ** 2)
    kernel = kernel / kernel.sum()
    k_h    = kernel.view(1, 1, ksize, 1).expand(C, 1, ksize, 1)
    k_w    = kernel.view(1, 1, 1, ksize).expand(C, 1, 1, ksize)
    pad    = r
    out = F.conv2d(field_f.view(B * C, 1, H, W),
                   kernel.view(1, 1, ksize, 1),
                   padding=(pad, 0), groups=1)
    out = F.conv2d(out,
                   kernel.view(1, 1, 1, ksize),
                   padding=(0, pad), groups=1)
    return out.view(B, C, H, W)


def multi_scale_hazing_loss(pred_full, gt_full, precip_ch=PRECIP_CH):
    p = pred_full[:, precip_ch:precip_ch + 1].float()
    g = gt_full[:,  precip_ch:precip_ch + 1].float()
    SCALES = [
        (1.0,  0.40),
        (4.0,  0.30),
        (12.0, 0.20),
        (32.0, 0.10),
    ]
    total = torch.tensor(0.0, device=pred_full.device)
    for sigma_px, w in SCALES:
        pb = _gaussian_blur_2d(p, sigma_px)
        gb = _gaussian_blur_2d(g, sigma_px)
        struct = F.huber_loss(pb, gb, delta=0.5)
        pf     = torch.fft.rfft2(p.squeeze(1))
        gf     = torch.fft.rfft2(g.squeeze(1))
        H, W_h = pf.shape[-2], pf.shape[-1]
        W_full  = (W_h - 1) * 2
        fy = torch.fft.fftfreq(H, device=p.device).abs()
        fx = torch.fft.rfftfreq(W_full, device=p.device).abs()
        freq = (fy[:, None]**2 + fx[None, :]**2).sqrt()
        f_centre = 1.0 / (2.0 * math.pi * max(sigma_px, 0.5))
        f_centre = min(f_centre, 0.49)
        bandpass = torch.exp(-0.5 * ((freq - f_centre) / (f_centre * 0.5 + 1e-4)) ** 2)
        psd_p    = (pf.abs() ** 2 * bandpass).mean()
        psd_g    = (gf.abs() ** 2 * bandpass).mean().clamp(min=1e-8)
        spectral = (torch.log(psd_p / psd_g + 1e-8)) ** 2
        total    = total + w * (struct + 0.5 * spectral)
    return total

def laplacian_loss(pred, target):
    # Standard 3x3 Laplacian Kernel
    kernel = torch.tensor([[[[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]]]], dtype=torch.float32).to(pred.device)
    
    # Apply filter to both pred and target
    lap_pred = F.conv2d(pred, kernel, padding=1)
    lap_target = F.conv2d(target, kernel, padding=1)
    
    # Minimize the difference between the "edge maps"
    return F.mse_loss(lap_pred, lap_target)
def sobel_loss(pred, target):

    pred_f   = pred.float()
    target_f = target.float()

    sobel_x = torch.tensor(
        [[[[-1,0,1],[-2,0,2],[-1,0,1]]]],
        device=pred.device,
        dtype=torch.float32
    )

    sobel_y = torch.tensor(
        [[[[-1,-2,-1],[0,0,0],[1,2,1]]]],
        device=pred.device,
        dtype=torch.float32
    )

    px = F.conv2d(pred_f, sobel_x, padding=1)
    py = F.conv2d(pred_f, sobel_y, padding=1)

    tx = F.conv2d(target_f, sobel_x, padding=1)
    ty = F.conv2d(target_f, sobel_y, padding=1)

    return F.l1_loss(px, tx) + F.l1_loss(py, ty)
    
    
def quantile_calibration_loss(pred, target):
    qs = [0.5, 0.7, 0.9, 0.95, 0.99]
    loss = 0.0
    for q in qs:
        p_q = torch.quantile(pred.view(pred.size(0), -1), q, dim=1)
        t_q = torch.quantile(target.view(target.size(0), -1), q, dim=1)
        loss += F.l1_loss(p_q, t_q)
    return loss / len(qs)

def mass_conservation_loss(pred, target):
    mass_pred = pred.sum(dim=[1, 2, 3])
    mass_gt   = target.sum(dim=[1, 2, 3])
    
    # Very soft relative mass error - much more training-friendly
    rel_error = torch.abs(mass_pred - mass_gt) / (mass_gt + 1e-6)
    return rel_error.mean() * 0.05     # very small weight

def gaussianity_loss(pred):
    # Encourages diffusion latent to stay near N(0,1) in log-space
    return (pred.mean().abs() + (pred.std() - 1.0).abs())
# =============================================================================
# COMBINED LOSS  (v4: SSIM added, phase weight raised, spectral reduced)
def corrdiff_loss_singapore(pred_full, y_fine_full,
                             residual_pred, residual_target,
                             mu, sigma, sigma_data,
                             wet_thresh_precip, c_skip, c_out,
                             weight_map=None, epoch=0):
    """
    Singapore-optimized loss function. 
    Fix: Removed UnboundLocalError by re-ordering variable assignments 
    and replacing explicit PCC with structural surrogates (SSIM + Gradient).
    """
    B = residual_pred.size(0)
    device = pred_full.device

    # --- 1. DEFINE PRECIPITATION FIELDS EARLY (Fixes UnboundLocalError) ---
    pred_p_full = pred_full[:, PRECIP_CH:PRECIP_CH+1]
    gt_p_full   = y_fine_full[:, PRECIP_CH:PRECIP_CH+1]
    
    # Residuals for precip channel
    r_pred_p = residual_pred[:,  PRECIP_CH:PRECIP_CH+1]
    r_tgt_p  = residual_target[:, PRECIP_CH:PRECIP_CH+1]

    # --- 2. EDM NOISE WEIGHTING ---
    snr_w = min_snr_weight(sigma, sigma_data, snr_clip=SNR_CLIP)
    snr_w = snr_w.reshape(-1, 1, 1, 1)

    lam = (sigma**2 + sigma_data**2) / (sigma * sigma_data + 1e-8)**2
    lam = lam.reshape(-1, 1, 1, 1)
    
    err = (residual_pred - residual_target)**2
    error_map = (pred_p_full - gt_p_full).abs().detach()
    
    
    adaptive_weight = 1.0 + 2.0 * (error_map / (error_map.mean() + 1e-6))
    adaptive_weight = adaptive_weight.clamp(1.0, 3.0)
    
    final_weight = weight_map * adaptive_weight
    
    if weight_map is not None:
        # Focus EDM loss using the Singapore weight map
        #edm_loss = (snr_w * lam * err * weight_map).sum() / (weight_map.sum() + 1e-8)
        edm_loss = (snr_w * lam * err * final_weight).sum() / (final_weight.sum() + 1e-8)
        mask = (weight_map > 1.5).float()
    else:
        edm_loss = (snr_w * lam * err).mean()
        mask = None

    # --- 3. STRUCTURAL & SPECTRAL LOSSES ---
    # Spectral curriculum: increases focus on high-freq details as training progresses
    spectral_loss = spectral_loss_weighted(residual_pred, residual_target)
    sobel_l = sobel_loss(pred_p_full, gt_p_full)
    if epoch < 30:
        spec_w = 0.0001
    elif epoch < 100:
        spec_w = 0.01
    else:
        spec_w = 0.02

    # Gradient Loss (Replaces Phase/PCC): Aligning edges aligns correlation
    ph_loss = fourier_phase_loss(pred_p_full, gt_p_full)
    rain_mask = (gt_p_full > wet_thresh_precip).float()

    wet_weight = 1.0 + 4.0 * rain_mask   # strong emphasis

    # SSIM Loss: Direct spatial structural similarity (The best surrogate for PCC)
    ssim_l = ssim_loss(pred_p_full, gt_p_full)

    # --- 4. PHYSICAL & STATISTICAL CONSTRAINTS ---
    # Sparsity: Penalizes rain in areas where the low-res/mu says it's dry
    dry_mask = (r_tgt_p + mu[:, PRECIP_CH:PRECIP_CH+1] < wet_thresh_precip).float()
    # 1. Identify pixels where GT is actually dry (The "Truth")
    true_dry_mask = (gt_p_full < wet_thresh_precip).float()
    
    # 2. Identify pixels where Coarse MU is wrongly wet (The "Ghost" area)
    # This finds exactly those "bridges" you mentioned
    ghost_bridge_mask = true_dry_mask * (mu[:, PRECIP_CH:PRECIP_CH+1] > wet_thresh_precip).float()
    
    # 3. Apply a "Bridge-Killer" penalty
    # We punish r_pred_p heavily if it tries to justify the coarse rain
    bridge_penalty = (F.relu(r_pred_p) * ghost_bridge_mask).pow(2).mean()
    
    # 4. Total Sparsity (General dry + Specific bridge killing)
    sparsity_loss = (F.relu(r_pred_p) * true_dry_mask).pow(2).mean() + 2.0 * bridge_penalty

    # Variance: Ensures the model doesn't "mute" the intensity of weather features
    c_out_w  = (c_out / (c_skip + c_out + 1e-8)).detach().squeeze()
    var_pred = r_pred_p.var(dim=(1, 2, 3))
    var_tgt  = r_tgt_p.var(dim=(1, 2, 3))
    var_loss = (c_out_w * (torch.abs(var_pred - var_tgt) / (var_tgt + 1e-6))).mean()

    # Tail/Percentile: Forces the model to get extreme weather (heavy rain) right
    if mask is not None:
        t_loss = region_tail_loss(r_pred_p, r_tgt_p, mask, k=20)
    else:
        #t_loss = tail_loss(r_pred_p, r_tgt_p, k=50)
        t_loss = percentile_loss(r_pred_p, r_tgt_p)
    
    pct_loss = percentile_loss(r_pred_p, r_tgt_p)

    # TV Loss: Edge smoothing/denoising
    tv_loss = (
        (r_pred_p[:, :, 1:, :] - r_pred_p[:, :, :-1, :]).abs().mean() +
        (r_pred_p[:, :, :, 1:] - r_pred_p[:, :, :, :-1]).abs().mean()
    )

    # Multi-scale Haze: Prevents "checkerboard" artifacts and spectral bias
    pred_full_det = mu.detach() + residual_pred
    haze_loss = multi_scale_hazing_loss(pred_full_det, y_fine_full)
    pcc_l = pcc_loss(pred_p_full, gt_p_full)
    mask = (weight_map > 1.5).float()
        
    sg_struct = ssim_loss(
        pred_p_full * final_weight,
        gt_p_full * final_weight
    )
    
    sg_pcc_l = pcc_loss(
        pred_p_full * final_weight,
        gt_p_full * final_weight
    )
    sg_loss = sg_struct + sg_pcc_l
    pred_max = pred_p_full.view(B, -1).max(dim=1).values
    gt_max   = gt_p_full.view(B, -1).max(dim=1).values
    
    peak_loss = torch.mean(
        torch.abs(pred_max - gt_max) / (gt_max + 1e-6)
    )
    quant_loss = quantile_calibration_loss(pred_p_full, gt_p_full)
    mass_loss  = mass_conservation_loss(pred_p_full, gt_p_full)
    gauss_loss = gaussianity_loss(r_pred_p)


    # Singapore-specific weighted MSE
    if weight_map is not None:
        sg_mse = region_weighted_loss(r_pred_p, r_tgt_p, weight_map)
    else:
        sg_mse = (r_pred_p.float() - r_tgt_p.float()).pow(2).mean()
    
    lap_l = laplacian_loss(pred_p_full, gt_p_full)
    # --- 5. DYNAMIC LAPLACIAN WEIGHT (Curriculum) ---
    # Start at 0, ramp up to 8.0 by Epoch 40
    if epoch < 20:
        lap_w = 0.0
    elif epoch < 40:
        # Linear ramp: (epoch - 10) * (8.0 / 30)
        lap_w = (epoch - 20) * 0.266
    else:
        lap_w = 8.0
    
    lap_l = laplacian_loss(pred_p_full, gt_p_full)
    
    loss = (
        1.5    * edm_loss +
        2.0    * pcc_l +
        1.2    * ssim_l +
        1.0    * t_loss +
        lap_w  * lap_l +
        1.2    * ph_loss +
        0.5    * haze_loss +
        0.2    * var_loss +
        0.5    * sparsity_loss +
        3.0    * peak_loss +
        3.0    * sg_loss +        # was 6.0
        2.0    * quant_loss +     # new: eliminates intensity bias
        0.5    * mass_loss +      # new: prevents ghost rain
        0.3    * gauss_loss +     # new: stabilizes latent distribution
        1.0    * sobel_l+
        spec_w * spectral_loss
    )
    return loss, {
        "edm": edm_loss.item(),
        "spectral": spectral_loss.item(),
        "phase": ph_loss.item(),
        "sparsity": sparsity_loss.item(),
        "var": var_loss.item(),
        "tail": t_loss.item(),
        "pct": pct_loss.item(),
        "tv": tv_loss.item(),
        "haze": haze_loss.item(),
        "ssim": ssim_l.item(),
        "sg_mse": sg_mse.item(),
        "quant": quant_loss.item(),
        "mass":  mass_loss.item(),
        "gauss": gauss_loss.item()
    }
# =============================================================================
# VALIDATION METRICS
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


def intensity_metrics(pred_p, target_p, wet_thresh):
    p_max = pred_p.reshape(pred_p.size(0), -1).max(dim=1).values
    t_max = target_p.reshape(target_p.size(0), -1).max(dim=1).values
    wet   = t_max > wet_thresh
    if wet.sum() == 0:
        return torch.tensor(0.0, device=pred_p.device)
    return torch.abs(p_max[wet] - t_max[wet]).mean()


def psd_ratio_metric(pred_p, target_p):
    pf = torch.fft.rfft2(pred_p.float()).abs().pow(2).mean()
    tf = torch.fft.rfft2(target_p.float()).abs().pow(2).mean().clamp(min=1e-8)
    return (pf / tf).item()


# =============================================================================
# [P6] EMA WITH WARMUP
# Ramps decay from 0.99 -> 0.9999 over first 1000 update steps.
# At step 0 with decay=0.9999 the shadow barely moves; at step=10 still only
# 0.01% of the new params make it through. Warmup ensures val sees a meaningful
# EMA from epoch 1.
# =============================================================================

class EMA:
    def __init__(self, model, decay_final=0.9995, warmup_steps=1000):
        self.decay_final  = decay_final
        self.warmup_steps = warmup_steps
        self.step         = 0
        self.shadow = {n: p.data.clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    def _decay(self):
        # Linear warmup: decay starts at 0.99 and ramps to decay_final
        frac  = min(self.step / max(self.warmup_steps, 1), 1.0)
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

    BATCH_SIZE    = 16
    LEARNING_RATE = 5e-5
    EPOCHS        = 15000
    PATIENCE      = 200
    P_STD         = 0.8
    N_STEPS_FAST  = 25
    N_STEPS_FULL  = 40
    N_ENSEMBLE    = 5

    # IMPORTANT: must match Dataset (log1p + normalization)
    

    

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset    = ClimateDataset(LR_FILES, HR_FILES, rank=rank)
    n          = len(dataset)
    train_size = int(0.65 * n)
    val_size   = int(0.15 * n)
    test_size  = n - train_size - val_size
    
    # IMPORTANT: must match Dataset (log1p + normalization)
    
    LOG_PRECIP_MEAN = dataset.mean["precip"]
    LOG_PRECIP_STD  = dataset.std["precip"]
    MAX_PRECIP_MM    = 300.0   # physical ceiling: no pixel can exceed 150mm/day
    NORM_CEILING     = (math.log1p(MAX_PRECIP_MM) - LOG_PRECIP_MEAN) / (LOG_PRECIP_STD + 1e-8)
    global _NORM_CEILING
    _NORM_CEILING = float(NORM_CEILING)
    log(f"  NORM_CEILING = {_NORM_CEILING:.4f}  (= {MAX_PRECIP_MM}mm in normalised space)")
    
    if rank == 0:
        precip_sample = dataset._hr_arrays["precip"].reshape(-1)
    
        N_SAMPLE = 1_000_000
        if precip_sample.size > N_SAMPLE:
            idx = np.random.choice(precip_sample.size, N_SAMPLE, replace=False)
            precip_sample = precip_sample[idx]
    
        WET_THRESH_NORM = np.quantile(precip_sample, 0.7)
    else:
        WET_THRESH_NORM = 0.0
        


    
    # broadcast to all GPUs
    wt_tensor = torch.tensor(WET_THRESH_NORM, device=device)
    dist.broadcast(wt_tensor, src=0)
    WET_THRESH_NORM = wt_tensor.item()

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, _ = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    log(f"Dataset: {n}  Train {train_size}  Val {val_size}  Test {test_size}")
    log(f"WET_THRESH_NORM = {WET_THRESH_NORM:.4f}")
    log(f"USE_PAPER_EMBEDDING = {USE_PAPER_EMBEDDING}")

    train_sampler = DistributedSampler(train_set, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_set, num_replicas=world_size,
                                       rank=rank, shuffle=False)
    train_loader  = DataLoader(train_set, batch_size=BATCH_SIZE,
                               sampler=train_sampler, num_workers=4,
                               pin_memory=True, persistent_workers=True)
    val_loader    = DataLoader(val_set, batch_size=BATCH_SIZE,
                               sampler=val_sampler, num_workers=2,
                               pin_memory=True, persistent_workers=True)

    # ------------------------------------------------------------------
    # Embedding framework
    # ------------------------------------------------------------------
    embedding_fw = None
    # ------------------------------------------------------------------
    # Build Singapore weight map (CORRECT PLACE)
    # ------------------------------------------------------------------
    _sample_hr = next(iter(train_loader))[1]
    HR_H = _sample_hr.shape[-2]
    HR_W = _sample_hr.shape[-1]
    del _sample_hr
    
    singapore_wmap = build_singapore_weight_map(
        HR_H, HR_W, dataset.lats, dataset.lons, device
    )
    
    log(f"Singapore weight map ready: {HR_H}x{HR_W}")
    if USE_PAPER_EMBEDDING:
        if rank == 0:
            s_freq = find_psd_intersection(train_loader, device, n_batches=80)
        else:
            s_freq = 0.0
        s_tensor = torch.tensor(s_freq, device=device)
        if world_size > 1:
            dist.broadcast(s_tensor, src=0)
        s_freq = s_tensor.item()

        embedding_fw = EmbeddingFramework(s_freq=s_freq, n_noise_steps=45,
                                          scale_factor=4, device=device)
        if rank == 0:
            log("  Collecting training data for QDM fitting...")
            lr_list, hr_list = [], []
            with torch.no_grad():
                for i, (lr_b, hr_b) in enumerate(train_loader):
                    if i >= 300:
                        break
                    lr_list.append(lr_b)
                    hr_list.append(hr_b)
            lr_all = torch.cat(lr_list)
            hr_all = torch.cat(hr_list)
            singapore_mask = (singapore_wmap.squeeze() > 1.5).cpu().numpy()
            
            embedding_fw.fit_qdm(
                lr_all, hr_all,
                wet_thresh=WET_THRESH_NORM,
                n_quantiles=N_QUANTILES,
                singapore_mask=singapore_mask
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

        log(f"  Embedding ready: s_freq={s_freq:.4f}  "
            f"noise_steps={embedding_fw._determine_noise_steps()}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    regressor = ClimateRegressor().to(device)
    regressor.load_state_dict(torch.load("regressor.pth", map_location=device))
    regressor.eval()

    diffusion_net = SingaporeCorrDiffNet(in_channels=12).to(device)

    model_core = SingaporeCorrDiffModel(
        diffusion_net, regressor,
        sigma_data=0.5, sigma_min=0.002, sigma_max=1.0, rho=_RHO
    ).to(device)

    log("Estimating sigma_data ...")
    residual_sigma_data = estimate_sigma_data_hybrid(
        model_core, embedding_fw, train_loader,
        device, WET_THRESH_NORM, n_batches=50
    )
    has_data = torch.tensor(1.0 if residual_sigma_data > 0.0 else 0.0, device=device)
    sigma_tensor = torch.tensor(residual_sigma_data, device=device)
    if world_size > 1:
        dist.all_reduce(sigma_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(has_data,     op=dist.ReduceOp.SUM)
    residual_sigma_data = (sigma_tensor / has_data.clamp(min=1.0)).item()

    sigma_min = residual_sigma_data * 0.01
    sigma_max = residual_sigma_data * 1.2
    P_mean    = math.log(residual_sigma_data) + 0.3

    model_core.sigma_data = residual_sigma_data
    model_core.sigma_min  = sigma_min
    model_core.sigma_max  = sigma_max

    log(f"  sigma_data={residual_sigma_data:.4f}  "
        f"sigma_min={sigma_min:.4f}  sigma_max={sigma_max:.4f}")
    log(f"  P_mean={P_mean:.4f}  P_std={P_STD}")

    # ------------------------------------------------------------------
    # [R] Build Singapore spatial weight map (once, cached on device)
    # ------------------------------------------------------------------

    log(f"  Singapore weight map: box={SINGAPORE_BOX}  "
        f"weight={SINGAPORE_WEIGHT}x  grid={HR_H}x{HR_W}")
    log(f"  Singapore region size: "
        f"{SINGAPORE_BOX[1]-SINGAPORE_BOX[0]} x "
        f"{SINGAPORE_BOX[3]-SINGAPORE_BOX[2]} pixels")

    # ------------------------------------------------------------------
    # DDP + optimiser
    # ------------------------------------------------------------------
    model_core.diffusion_net = DDP(model_core.diffusion_net,
                                   device_ids=[local_rank],
                                   output_device=local_rank,
                                   find_unused_parameters=False)

    diffusion_net_train = model_core.diffusion_net

    # [P6] EMA with warmup over first 1000 steps
    ema       = EMA(diffusion_net_train, decay_final=0.9995, warmup_steps=1000)
    optimizer = AdamW(diffusion_net_train.parameters(),
                      lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS,
                                  eta_min=LEARNING_RATE * 0.01)
    scaler    = GradScaler("cuda")

    BEST_CKPT_PATH   = os.path.join(SAVE_DIR, "corrdiff_singapore_v4.pth")
    LATEST_CKPT_PATH = os.path.join(SAVE_DIR, "corrdiff_singapore_v4_latest.pth")

    best_metric      = float("inf")
    patience_counter = 0
    loss_keys = ['edm', 'spectral', 'phase', 'sparsity', 'var', 'tail', 'pct',
                 'tv', 'haze', 'ssim', 'sg_mse', 'quant', 'mass', 'gauss',
                 'pcc']   # expanded   # add three

    log("")
    log("=" * 70)
    log("CorrDiff Singapore v4: targeting PCC > 0.95 + Singapore region focus")
    log("=" * 70)
    log("v4 additions: SSIM 1.5 | Phase thresh 0.05 | 5% cond dropout")
    log("              Spectral 0.08 | Phase weight 0.8 | EMA warmup 1000 steps")
    log(f"[R] Singapore focus: box={SINGAPORE_BOX} weight={SINGAPORE_WEIGHT}x sg_mse 2.0")
    log("Loss: EDM 1.0 | Spec 0.08 | Spar 0.05 | Var 0.2 | Tail 1.0")
    log("      TV 0.025 | Pct 0.8 | Phase 0.8 | Haze 0.3 | SSIM 1.5 | SgMSE 2.0")
    log(f"Val: fast {N_STEPS_FAST} / full {N_STEPS_FULL} | ensemble N={N_ENSEMBLE}")
    log("=" * 70)
    log("")

    # =========================================================================
    # EPOCH LOOP
    # =========================================================================
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        model_core.diffusion_net.train()
        train_loss   = 0.0
        loss_details = {k: 0.0 for k in loss_keys}
        n_batches    = 0
        t0           = time.time()

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
                pred, mu, res_pred, res_target, c_skip, c_out = \
                    model_core.forward_train(hr_b, lr_b, sigma,
                                            embedding_fw=embedding_fw)
                loss, metrics = corrdiff_loss_singapore(
                    pred, hr_b,
                    res_pred, res_target,
                    mu, sigma,
                    residual_sigma_data, 
                    WET_THRESH_NORM,
                    c_skip, c_out,
                    weight_map=singapore_wmap,
                    epoch=epoch
                )

            if torch.isnan(loss) or torch.isinf(loss):
                log(f"[W] NaN/Inf at epoch {epoch+1}, skipping batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion_net_train.parameters(),
                                           max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(diffusion_net_train)

            train_loss += loss.item()
            for k in loss_keys:
                loss_details[k] += metrics.get(k, 0.0)
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # =====================================================================
        # VALIDATION
        # =====================================================================
        run_val  = (epoch < 10) or (epoch % 5 == 0)
        full_val = (epoch < 10) or (epoch % 25 == 0)
        n_infer  = N_STEPS_FULL if full_val else N_STEPS_FAST

        if run_val:
            model_core.diffusion_net.eval()
            val_pcc_s1 = val_pcc_s2 = val_int_s1 = val_int_s2 = 0.0
            val_psd = val_phase = val_spread = 0.0
            # [R] Singapore-specific metrics
            val_sg_pcc_s1 = val_sg_pcc_s2 = 0.0
            val_sg_int_s1 = val_sg_int_s2 = 0.0
            n_val = 0

            with torch.no_grad():
                ema.backup(diffusion_net_train)
                ema.apply_shadow(diffusion_net_train)

                for lr_b, hr_b in val_loader:
                    lr_b = lr_b.to(device)
                    hr_b = hr_b.to(device)

                    p_max = hr_b[:, PRECIP_CH].reshape(hr_b.size(0), -1).max(dim=1).values
                    wet   = p_max > WET_THRESH_NORM
                    if wet.sum() == 0:
                        continue

                    lr_w = lr_b[wet]
                    hr_w = hr_b[wet]

                    H, W   = hr_w.shape[-2], hr_w.shape[-1]
                    mu_raw = model_core.get_mu(lr_w)
                    mu     = F.interpolate(mu_raw, size=(H, W),
                                           mode="bilinear", align_corners=False)

                    samples = torch.stack([
                        model_core.sample(lr_w, n_steps=n_infer,embedding_fw=embedding_fw,weight_map=singapore_wmap)
                        for _ in range(N_ENSEMBLE)
                    ], dim=0)
                    pred_s2        = samples.mean(dim=0)
                    pred_s2_spread = torch.nan_to_num(samples.std(dim=0)).mean().item()

                    mu_p  = mu[:, PRECIP_CH]
                    s2_p  = pred_s2[:, PRECIP_CH]
                    gt_p  = hr_w[:, PRECIP_CH]

                    pcc_s1, _  = wet_pcc_batch(mu_p, gt_p, WET_THRESH_NORM)
                    pcc_s2, _  = wet_pcc_batch(s2_p, gt_p, WET_THRESH_NORM)
                    int_err_s1 = intensity_metrics(mu_p, gt_p, WET_THRESH_NORM)
                    int_err_s2 = intensity_metrics(s2_p, gt_p, WET_THRESH_NORM)
                    psd_r      = psd_ratio_metric(s2_p, gt_p)
                    phase_r    = phase_coherence_metric(
                        pred_s2[:, PRECIP_CH:PRECIP_CH+1] - mu[:, PRECIP_CH:PRECIP_CH+1],
                        hr_w[:,   PRECIP_CH:PRECIP_CH+1]  - mu[:, PRECIP_CH:PRECIP_CH+1]
                    )

                    val_pcc_s1 += pcc_s1.item()
                    val_pcc_s2 += pcc_s2.item()
                    val_int_s1 += int_err_s1.item()
                    val_int_s2 += int_err_s2.item()
                    val_psd    += psd_r
                    val_phase  += phase_r
                    val_spread += pred_s2_spread

                    # [R] Singapore-specific PCC and intensity
                    sg_pcc_s1 = region_pcc_metric(mu_p,  gt_p, SINGAPORE_BOX)
                    sg_pcc_s2 = region_pcc_metric(s2_p,  gt_p, SINGAPORE_BOX)
                    sg_int_s1 = region_intensity_metric(mu_p, gt_p, SINGAPORE_BOX)
                    sg_int_s2 = region_intensity_metric(s2_p, gt_p, SINGAPORE_BOX)
                    val_sg_pcc_s1 += sg_pcc_s1
                    val_sg_pcc_s2 += sg_pcc_s2
                    val_sg_int_s1 += sg_int_s1
                    val_sg_int_s2 += sg_int_s2

                    n_val += 1

            ema.restore(diffusion_net_train)

            if n_val > 0:
                avg_pcc_s1     = val_pcc_s1 / n_val
                avg_pcc_s2     = val_pcc_s2 / n_val
                avg_int_err_s1 = val_int_s1 / n_val
                avg_int_err_s2 = val_int_s2 / n_val
                avg_psd        = val_psd    / n_val
                avg_phase      = val_phase  / n_val
                avg_spread     = val_spread / n_val

                # [R] Singapore averages
                avg_sg_pcc_s1 = val_sg_pcc_s1 / n_val
                avg_sg_pcc_s2 = val_sg_pcc_s2 / n_val
                avg_sg_int_s1 = val_sg_int_s1 / n_val
                avg_sg_int_s2 = val_sg_int_s2 / n_val

                dpcc        = avg_pcc_s2 - avg_pcc_s1
                sg_dpcc     = avg_sg_pcc_s2 - avg_sg_pcc_s1
                int_gain    = avg_int_err_s1 - avg_int_err_s2
                sg_int_gain = avg_sg_int_s1  - avg_sg_int_s2
                psd_penalty = abs(math.log(max(avg_psd, 1e-6)))
                phase_pen   = 1.0 - avg_phase

                # [R] val_metric now uses Singapore PCC and intensity as primary signals.
                # Global PCC still included with lower weight so global quality is maintained.
                val_metric = (
                    -0.85 * sg_dpcc           # Singapore PCC improvement (primary)
                    -0.15 * dpcc              # global PCC (secondary)
                    + 0.8 * avg_sg_int_s2    # Singapore intensity error vs GT
                    + 0.1 * avg_int_err_s2   # global intensity error vs GT
                    + 0.2 * psd_penalty
                    + 0.22 * phase_pen
                )

                val_tag = "FULL" if full_val else "FAST"
                avgs    = {k: loss_details[k] / max(n_batches, 1) for k in loss_keys}
                log(
                    f"Epoch {epoch+1:04d} [{val_tag}] | "
                    f"Loss {avg_train:.4f} "
                    f"[edm {avgs['edm']:.3f} "
                    f"spec {avgs['spectral']:.3f} "
                    f"spar {avgs['sparsity']:.4f} "
                    f"var {avgs['var']:.3f} "
                    f"tail {avgs['tail']:.3f} "
                    f"haze {avgs['haze']:.3f} "
                    f"pct {avgs['pct']:.3f} "
                    f"ph {avgs['phase']:.3f} "
                    f"ssim {avgs['ssim']:.3f} "
                    f"sg {avgs['sg_mse']:.3f}]\n"
                    f"          | [GLOBAL] S1 PCC {avg_pcc_s1:.4f} "
                    f"| S2 PCC {avg_pcc_s2:.4f} "
                    f"| dPCC {dpcc:+.5f}\n"
                    f"          | [GLOBAL] S1 IntErr {avg_int_err_s1:.4f} "
                    f"| S2 IntErr {avg_int_err_s2:.4f} "
                    f"| IntGain {int_gain:+.4f}\n"
                    f"          | [SINGAPORE] S1 PCC {avg_sg_pcc_s1:.4f} "
                    f"| S2 PCC {avg_sg_pcc_s2:.4f} "
                    f"| dPCC {sg_dpcc:+.5f}\n"
                    f"          | [SINGAPORE] S1 IntErr {avg_sg_int_s1:.4f} "
                    f"| S2 IntErr {avg_sg_int_s2:.4f} "
                    f"| IntGain {sg_int_gain:+.4f}\n"
                    f"          | PSD {avg_psd:.3f} | steps {n_infer} | ens {N_ENSEMBLE} "
                    f"| Phase {avg_phase:.4f} | PhPen {phase_pen:.4f} "
                    f"| Spread {avg_spread:.4f} "
                    f"| EMA_decay {ema._decay():.5f} "
                    f"| ValMet {val_metric:.5f}"
                )

                ckpt = {
                    "epoch":            epoch,
                    "model_state_dict": model_core.diffusion_net.module.state_dict(),
                    "sigma_data":       residual_sigma_data,
                    "sigma_min":        sigma_min,
                    "sigma_max":        sigma_max,
                    "wet_thresh_norm":  WET_THRESH_NORM,
                    "precip_mean": LOG_PRECIP_MEAN,
                    "precip_std":  LOG_PRECIP_STD,
                    "use_embedding":    USE_PAPER_EMBEDDING,
                    "embedding_state":  embedding_fw.state_dict() if embedding_fw else None,
                    "singapore_box":    SINGAPORE_BOX,
                    "singapore_weight": SINGAPORE_WEIGHT,
                    # Global metrics
                    "pcc_s1":       avg_pcc_s1,
                    "pcc_s2":       avg_pcc_s2,
                    "dpcc":         dpcc,
                    "int_err_s1":   avg_int_err_s1,
                    "int_err_s2":   avg_int_err_s2,
                    "int_gain":     int_gain,
                    "psd_ratio":    avg_psd,
                    "phase":        avg_phase,
                    # Singapore-specific metrics
                    "sg_pcc_s1":    avg_sg_pcc_s1,
                    "sg_pcc_s2":    avg_sg_pcc_s2,
                    "sg_dpcc":      sg_dpcc,
                    "sg_int_err_s1": avg_sg_int_s1,
                    "sg_int_err_s2": avg_sg_int_s2,
                    "sg_int_gain":  sg_int_gain,
                    "val_metric":   val_metric,
                }

                if rank == 0:
                    torch.save(ckpt, LATEST_CKPT_PATH)
                    log("  Saved LATEST")

                if full_val:
                    EPS = 1e-4
                    if val_metric < best_metric - EPS:
                        best_metric      = val_metric
                        patience_counter = 0
                        if rank == 0:
                            torch.save(ckpt, BEST_CKPT_PATH)
                            log("  Saved BEST")
                    else:
                        patience_counter += 1
                        if patience_counter >= PATIENCE:
                            log(f"Early stopping at epoch {epoch+1} "
                                f"({PATIENCE} full-val rounds without improvement)")
                            break

        elapsed = round(time.time() - t0, 1)
        if rank == 0 and not run_val:
            log(f"Epoch {epoch+1:04d} | Train: {avg_train:.4f} | Time: {elapsed}s")

    cleanup_ddp()
    log("Training complete.")


if __name__ == "__main__":
    main()
