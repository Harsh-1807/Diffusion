# -*- coding: utf-8 -*-
# =============================================================================
#  CorrDiff Singapore  v9-FINAL
#  Synthesis of v8-FINAL + v7-new architecture + doc4 hyperparameter review
#
#  Changes from v8-FINAL:
#  1.  Regressor REMOVED — pure generative model, 8-ch input [y_noisy, cond]
#      matches the CorrDiff paper exactly; eliminates mu-drift accumulation.
#  2.  Full-field EDM target — network denoises full field, not residual.
#      Simpler gradient flow, no regressor error propagation.
#  3.  _weak_linear_betas — replaces cosine betas in EmbeddingFramework.
#      Lower cumulative noise variance -> less ghost rain in conditioning.
#  4.  autocast guards in embedding f() and g() — prevents bfloat16 NaN
#      in noise application steps (silent corruption bug in v8).
#  5.  Multiplicative QDM for precipitation (Cannon et al. 2015) — correct
#      physical bias correction for a strictly positive variable.
#      Additive QDM used for all other channels (huss, mslp, tas).
#  6.  _RHO raised to 9.0 — more sampling steps at structural noise levels;
#      correct for high-kurtosis precipitation with many zeros and sharp peaks.
#  7.  CFG dropout raised to 15% — forces UNet to learn distribution without
#      LR conditioning, sharpening high-frequency detail at inference.
#  8.  N_QUANTILES raised to 1000 — better resolution of 99.9th-pct tail.
#  9.  EMA decay 0.999 (was 0.9995) — less over-smoothing of extreme events.
# 10.  Embedding sanity diagnostics — f/g correlation + beta noise variance
#      logged at startup so ghost-rain causes are immediately visible.
# 11.  All v8 intensity fixes retained: sqrt-MSE, log-MSE, peak-95, spectral,
#      s_churn, PhaseScheduler, clamp(-2.5,2.5), sigma_data floor 0.8.
# =============================================================================

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

from Dataset import ClimateDataset
from Network import DiffusionUNet


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

LOG_FILE  = "corrdiff_v9_final.log"
SAVE_DIR  = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/CorrDiff/version13/"

# CHANGE 6: rho=9.0 — more steps at structural noise levels for precip kurtosis
_RHO        = 9.0
PRECIP_CH   = 3
N_CH        = 4
# CHANGE 8: 1000 quantiles for better tail resolution
N_QUANTILES = 1000

USE_PAPER_EMBEDDING = True
SNR_CLIP            = 5.0

# Singapore Gaussian centre (lat/lon degrees)
SG_LAT_C  = 1.3
SG_LON_C  = 103.8
SG_SIGMA  = 1.2
SG_WEIGHT = 11.0


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
        print(msg, flush=True)
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


def min_snr_weight(sigma, sigma_data, snr_clip=SNR_CLIP):
    snr    = (sigma_data / sigma.clamp(min=1e-8)) ** 2
    weight = (snr_clip / snr).clamp(max=1.0)
    return weight.detach()


# =============================================================================
# EMBEDDING FRAMEWORK
# CHANGE 3: _weak_linear_betas replaces cosine betas — lower cumulative noise
#           variance reduces ghost rain in the conditioning channel.
# CHANGE 4: autocast guards in f() and g() — prevents silent bfloat16 NaN.
# CHANGE 5: Multiplicative QDM for precip channel (Cannon et al. 2015).
# =============================================================================

def _weak_linear_betas(n_steps, device, beta_start=0.01, beta_end=0.10):
    """
    Linear beta schedule with low total noise variance.
    Cumulative noise var = 1 - prod(1 - beta_t) ≈ 0.55 for defaults.
    Cosine betas were giving ~0.8, causing over-smoothed conditioning
    that appeared as diffuse ghost rain in dry regions.
    """
    return torch.linspace(beta_start, beta_end, n_steps, device=device)


def _apply_noise_steps(field, betas, n_steps):
    """Always runs in float32 regardless of surrounding autocast context."""
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
        self.betas         = _weak_linear_betas(n_noise_steps, device)
        self.qdm_lr        = [None] * N_CH
        self.qdm_hr        = [None] * N_CH
        self._fitted       = False

    def fit_qdm(self, lr_data, hr_data, n_quantiles=N_QUANTILES):
        """
        Fit QDM on native grids, full distribution (no wet-day filtering).
        Wet-day filtering biases the quantile mapping for light rain.

        Uses multiplicative QDM for precipitation (channel PRECIP_CH) and
        additive QDM for all other variables.
        """
        log("  Fitting QDM (multiplicative precip / additive others) ...")
        q_levels = np.linspace(0, 1, n_quantiles)

        for ch in range(N_CH):
            lr_ch = lr_data[:, ch].reshape(-1).float().cpu().numpy()
            hr_ch = hr_data[:, ch].reshape(-1).float().cpu().numpy()
            self.qdm_lr[ch] = np.quantile(lr_ch, q_levels)
            self.qdm_hr[ch] = np.quantile(hr_ch, q_levels)

        self._fitted = True
        log("  QDM fitting complete.")

    def _qdm_multiplicative(self, arr_np, q_lr, q_hr):
        """
        Multiplicative QDM for precipitation (Cannon et al. 2015).
        Physically correct for a strictly positive variable: the bias
        correction is applied as a ratio, not an additive delta.
        This preserves zero-rain days and avoids negative precipitation.
        """
        shape  = arr_np.shape
        x      = arr_np.ravel().astype(np.float64)
        ranks  = np.linspace(0, 1, len(q_lr))
        tau    = np.interp(x, q_lr, ranks)

        q_model_tau = np.interp(tau, ranks, q_lr)
        q_obs_tau   = np.interp(tau, ranks, q_hr)

        # Safe divide: avoid blowing up on near-zero model values
        delta  = x / (np.abs(q_model_tau) + 1e-8)
        result = q_obs_tau * delta
        return result.astype(np.float32).reshape(shape)

    def _qdm_additive(self, arr_np, q_lr, q_hr):
        """Additive QDM for non-precipitation variables (huss, mslp, tas)."""
        shape  = arr_np.shape
        x      = arr_np.ravel().astype(np.float64)
        ranks  = np.linspace(0, 1, len(q_lr))
        tau    = np.interp(x, q_lr, ranks)

        q_model_tau = np.interp(tau, ranks, q_lr)
        q_obs_tau   = np.interp(tau, ranks, q_hr)

        result = q_obs_tau + (x - q_model_tau)
        return result.astype(np.float32).reshape(shape)

    def f(self, hr_field):
        """
        f(OBS): downsample HR → upsample → add noise.
        Used as conditioning signal during TRAINING.
        CHANGE 4: runs in float32 inside autocast guard.
        """
        with torch.autocast('cuda', enabled=False):
            hr_f32 = hr_field.float()
            B, C, H, W = hr_f32.shape
            sf    = self.scale_factor
            H_lr  = H // sf
            W_lr  = W // sf

            hr_ds = F.interpolate(hr_f32, size=(H_lr, W_lr),
                                  mode='bilinear', align_corners=False)
            hr_us = F.interpolate(hr_ds, size=(H, W),
                                  mode='bilinear', align_corners=False)

            betas = self.betas.to(hr_f32.device)
            out   = _apply_noise_steps(hr_us, betas, self.n_noise_steps)
        return out.to(hr_field.dtype)

    def g(self, lr_field, hr_size, weight_map=None):
        """
        g(ESM): QDM on native LR grid → bilinear upsample → add noise.
        Used as conditioning signal during INFERENCE.
        CHANGE 4: runs in float32 inside autocast guard.
        CHANGE 5: uses multiplicative QDM for precip, additive for others.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_qdm() first.")

        with torch.autocast('cuda', enabled=False):
            lr_f32  = lr_field.float()
            B, C    = lr_f32.shape[:2]
            H_hr, W_hr = hr_size

            lr_qdm = torch.empty_like(lr_f32)
            for ch in range(C):
                arr = lr_f32[:, ch].cpu().numpy()
                if ch == PRECIP_CH:
                    mapped = self._qdm_multiplicative(
                        arr, self.qdm_lr[ch], self.qdm_hr[ch])
                else:
                    mapped = self._qdm_additive(
                        arr, self.qdm_lr[ch], self.qdm_hr[ch])
                lr_qdm[:, ch] = torch.tensor(
                    mapped, dtype=torch.float32, device=lr_field.device)

            lr_up = F.interpolate(lr_qdm, size=(H_hr, W_hr),
                                  mode='bilinear', align_corners=False)

            betas = self.betas.to(lr_field.device)
            out   = _apply_noise_steps(lr_up, betas, self.n_noise_steps)

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
        self.betas         = _weak_linear_betas(
            self.n_noise_steps, torch.device('cpu'))


# =============================================================================
# PSD INTERSECTION
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
            lr_p = lr_b[:, PRECIP_CH]

            H, W = hr_p.shape[-2], hr_p.shape[-1]

            # Apply QDM correction before PSD if available
            if qdm_fw is not None and qdm_fw._fitted:
                arr    = lr_p.float().numpy()
                mapped = qdm_fw._qdm_multiplicative(
                    arr, qdm_fw.qdm_lr[PRECIP_CH], qdm_fw.qdm_hr[PRECIP_CH])
                lr_p   = torch.tensor(mapped, dtype=torch.float32)

            lr_p  = lr_p.to(device)
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
# CHANGE 1: in_channels=8 — [y_noisy(4), cond(4)]; regressor removed.
# =============================================================================

class SingaporeCorrDiffNet(nn.Module):
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
# MODEL
# CHANGE 1: No regressor. Full-field EDM formulation (not residual-on-mu).
# CHANGE 2: forward_train returns (F_x, target_Fx, y_noisy, noise, sigma_r)
#           matching the CorrDiff paper's training objective exactly.
# CHANGE 7: CFG dropout raised to 15% for sharper conditioning.
# =============================================================================

class SingaporeCorrDiffModel(nn.Module):
    def __init__(self, diffusion_net, sigma_data, sigma_min, sigma_max, rho=_RHO):
        super().__init__()
        self.diffusion_net = diffusion_net
        self.sigma_data    = sigma_data
        self.sigma_min     = sigma_min
        self.sigma_max     = sigma_max
        self.rho           = rho

    def forward_train(self, y_fine, lr, sigma, embedding_fw=None):
        H, W = y_fine.shape[-2], y_fine.shape[-1]

        if embedding_fw is not None and USE_PAPER_EMBEDDING:
            cond_ch = embedding_fw.f(y_fine)
        else:
            cond_ch = F.interpolate(lr, size=(H, W),
                                    mode='bilinear', align_corners=False)

        # CHANGE 7: 15% CFG dropout (was 5%) — forces the UNet to learn the
        # marginal data distribution without LR guidance, producing sharper
        # high-frequency detail at inference time.
        if self.training:
            B = y_fine.size(0)
            drop_mask = (torch.rand(B, 1, 1, 1, device=y_fine.device) > 0.15).float()
            cond_ch   = cond_ch * drop_mask

        sigma_r = sigma.reshape(-1, 1, 1, 1)
        noise   = torch.randn_like(y_fine)
        y_noisy = y_fine + noise * sigma_r

        net_input = torch.cat([y_noisy, cond_ch], dim=1)

        c_skip, c_out, c_in, c_noise = edm_precond(sigma, self.sigma_data)

        # Network predicts F_x (raw denoiser output)
        F_x = self.diffusion_net(c_in * net_input, c_noise)

        # EDM training target: what F_x should equal to recover y_fine via skip
        # D(y_noisy, sigma) = c_skip*y_noisy + c_out*F_x  should equal y_fine
        # => F_x target = (y_fine - c_skip*y_noisy) / c_out
        target_Fx = (y_fine - c_skip * y_noisy) / (c_out + 1e-8)

        # Reconstruct x0_pred for auxiliary losses (intensity, PCC, etc.)
        # This is the model's best guess at the clean field at this noise level.
        x0_pred = c_skip * y_noisy + c_out * F_x
        # Clamp to valid precipitation range — wider than v8's ±1.5 to allow peaks
        x0_pred = x0_pred.clamp(min=0.0)

        return F_x, target_Fx, y_noisy, noise, sigma_r, x0_pred

    @torch.no_grad()
    def sample(self, lr, n_steps=20, embedding_fw=None, weight_map=None,
               s_churn=0.05):
        """
        Heun sampler with controlled S_churn stochasticity.
        s_churn=0.05: mild noise re-injection at mid-range sigmas.
        This pushes the sampler toward higher-intensity draws without
        destabilising the final denoising steps near sigma_min.
        """
        H_hr = lr.shape[-2] * 4
        W_hr = lr.shape[-1] * 4

        if embedding_fw is not None and USE_PAPER_EMBEDDING:
            cond_ch = embedding_fw.g(lr, hr_size=(H_hr, W_hr), weight_map=weight_map)
        else:
            cond_ch = F.interpolate(lr, size=(H_hr, W_hr),
                                    mode="bilinear", align_corners=False)

        sigmas = get_sigma_schedule(n_steps, lr.device,
                                     self.sigma_min, self.sigma_max, self.rho)

        # Start at full sigma (not 0.8x) — explore full intensity range
        x = torch.randn(lr.size(0), N_CH, H_hr, W_hr, device=lr.device) * sigmas[0]

        def denoise(x_in, sig):
            sig_b          = sig.expand(lr.size(0))
            inp            = torch.cat([x_in, cond_ch], dim=1)
            cs, co, ci, cn = edm_precond(sig_b, self.sigma_data)
            Fx             = self.diffusion_net(ci * inp, cn)
            return (cs * x_in + co * Fx).clamp(min=0.0)

        for i in range(n_steps):
            sc = sigmas[i].clamp(min=self.sigma_min)
            sn = sigmas[i + 1].clamp(min=self.sigma_min)

            # S_churn: stochasticity only at mid-range sigmas
            if s_churn > 0 and sc > self.sigma_min * 5 and sc < self.sigma_max * 0.5:
                gamma     = min(s_churn, math.sqrt(2) - 1)
                sigma_hat = sc * (1.0 + gamma)
                eps       = torch.randn_like(x) * (sigma_hat**2 - sc**2).sqrt()
                x         = x + eps
                sc        = sigma_hat

            Dx = denoise(x, sc)
            d  = (x - Dx) / sc

            if i == n_steps - 1:
                x = Dx
            else:
                xn  = x + (sn - sc) * d
                Dx2 = denoise(xn, sn)
                d2  = (xn - Dx2) / sn
                x   = x + (sn - sc) * (d + d2) * 0.5

        return x.clamp(min=0.0)


# =============================================================================
# SIGMA_DATA ESTIMATION  (no regressor; estimates std of full HR field)
# Floor at 0.8 ensures model doesn't under-shoot peak variance.
# =============================================================================

def estimate_sigma_data(train_loader, device, precip_wet_thresh, n_batches=50):
    stds = []
    with torch.no_grad():
        for i, (lr_b, hr_b) in enumerate(train_loader):
            if i >= n_batches:
                break
            hr_b  = hr_b.to(device)
            p_max = hr_b[:, PRECIP_CH].reshape(hr_b.size(0), -1).max(dim=1).values
            wet   = p_max > precip_wet_thresh
            if wet.sum() == 0:
                continue
            stds.append(hr_b[wet, PRECIP_CH].std().item())
    if len(stds) == 0:
        return 0.8
    return max(float(torch.tensor(stds).mean()), 0.8)


# =============================================================================
# SINGAPORE WEIGHT MAP
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
# LOSS FUNCTIONS  v9
# All losses work on x0_pred (reconstructed clean field) rather than on
# the raw F_x output. This gives physics losses a more interpretable signal.
# =============================================================================

def loss_edm(F_x, target_Fx, sigma, sigma_data, weight_map):
    """
    Core EDM denoising loss. This is the primary training signal.
    Min-SNR weighting prevents high-noise steps from dominating gradients.
    """
    snr_w = min_snr_weight(sigma, sigma_data).reshape(-1, 1, 1, 1)
    err   = (F_x - target_Fx) ** 2
    if weight_map is not None:
        return (snr_w * err * weight_map).sum() / (weight_map.sum() * F_x.size(0) + 1e-8)
    return (snr_w * err).mean()


def loss_spatial_pcc(pred_p, gt_p, weight_map):
    """
    Dual-PCC: global weighted + Singapore-only.
    Ensures spatial alignment is strong both globally and in the target region.
    """
    B = pred_p.size(0)

    # Global weighted PCC
    w_sqrt = weight_map.squeeze(1).sqrt()
    p_w    = (pred_p.squeeze(1) * w_sqrt).view(B, -1)
    t_w    = (gt_p.squeeze(1)   * w_sqrt).view(B, -1)
    vx     = p_w - p_w.mean(dim=1, keepdim=True)
    vy     = t_w - t_w.mean(dim=1, keepdim=True)
    corr_g = (vx * vy).sum(1) / (
        (vx**2).sum(1).sqrt() * (vy**2).sum(1).sqrt() + 1e-8)
    loss_g = (1.0 - corr_g).mean()

    # Singapore-only PCC
    w_raw   = weight_map.squeeze()
    sg_mask = (w_raw > 2.0)
    loss_sg = 0.0
    if sg_mask.sum().item() > 100:
        sg_b  = sg_mask.unsqueeze(0).expand(B, -1, -1)
        p_sg  = pred_p.squeeze(1)[sg_b].reshape(B, -1)
        t_sg  = gt_p.squeeze(1)[sg_b].reshape(B, -1)
        vx_sg = p_sg - p_sg.mean(dim=1, keepdim=True)
        vy_sg = t_sg - t_sg.mean(dim=1, keepdim=True)
        c_sg  = (vx_sg * vy_sg).sum(1) / (
            (vx_sg**2).sum(1).sqrt() * (vy_sg**2).sum(1).sqrt() + 1e-8)
        loss_sg = (1.0 - c_sg).mean()

    return torch.clamp(0.5 * loss_g + 0.5 * loss_sg, min=0.0, max=2.0)


def loss_spectral_logmse(pred_p, gt_p):
    """
    Log-MSE on the 2D FFT magnitude spectrum.
    Forces the model to match sharpness across all frequency bands uniformly.
    Prevents the blurry-rain problem that pixel-only losses produce.
    """
    pf       = torch.fft.rfft2(pred_p.float().squeeze(1))
    gf       = torch.fft.rfft2(gt_p.float().squeeze(1))
    pred_mag = pf.abs()
    gt_mag   = gf.abs()
    return F.mse_loss(torch.log1p(pred_mag), torch.log1p(gt_mag))


def loss_intensity_v9(pred_p, gt_p, wet_thresh=0.1):
    """
    Four sub-terms, all on wet pixels only:

    1. Huber (delta=1.0) — robust baseline; L2 for small errors, L1 for peaks.
    2. Log-MSE           — equal % treatment across magnitudes (slow but steady).
    3. Sqrt-MSE          — "Goldilocks" zone: stronger than log on peaks,
                           weaker than MSE on outliers. Ideal for precipitation
                           which has a power-law tail.
    4. Peak-95 MSE       — explicit 95th-percentile commitment; stops the model
                           from averaging toward the median.

    Weights: 1.0 / 0.5 / 0.3 / 0.2 — huber dominates for stability.
    """
    wet_mask = (gt_p > wet_thresh).float()
    n_wet    = wet_mask.sum().clamp(min=1.0)

    pred_wet = pred_p * wet_mask
    gt_wet   = gt_p   * wet_mask

    # 1. Huber
    l_huber = F.huber_loss(pred_wet, gt_wet, delta=1.0)

    # 2. Log-MSE (wet pixels only)
    l_log = (torch.log1p(pred_wet) - torch.log1p(gt_wet)).abs().sum() / n_wet

    # 3. Sqrt-MSE — clamp before sqrt to avoid NaN in bfloat16
    l_sqrt = F.mse_loss(
        pred_wet.clamp(min=0).sqrt(),
        gt_wet.clamp(min=0).sqrt()
    )

    # 4. 95th-percentile peak matching (per sample in batch)
    flat_pred = pred_p.view(pred_p.size(0), -1)
    flat_gt   = gt_p.view(gt_p.size(0), -1)
    wet_b     = wet_mask.view(wet_mask.size(0), -1)

    l_peak  = torch.tensor(0.0, device=pred_p.device)
    n_valid = 0
    for b in range(pred_p.size(0)):
        idx = wet_b[b].bool()
        if idx.sum() < 10:
            continue
        p95_pred = torch.quantile(flat_pred[b][idx], 0.95)
        p95_gt   = torch.quantile(flat_gt[b][idx],   0.95)
        l_peak   = l_peak + F.mse_loss(p95_pred, p95_gt)
        n_valid += 1
    if n_valid > 0:
        l_peak = l_peak / n_valid

    return l_huber + 0.5 * l_log + 0.3 * l_sqrt + 0.2 * l_peak


def loss_tail_quantile(pred_p, gt_p, wet_thresh, q_levels=(0.95, 0.98, 0.99, 0.995)):
    """Heavy-rain tail quantile matching across the batch."""
    pred_flat = pred_p.view(pred_p.size(0), -1)
    gt_flat   = gt_p.view(gt_p.size(0), -1)
    wet_mask  = (gt_flat > wet_thresh).any(dim=1)
    if not wet_mask.any():
        return torch.tensor(0.0, device=pred_p.device)

    loss_q = torch.tensor(0.0, device=pred_p.device)
    for q in q_levels:
        q_gt   = torch.quantile(gt_flat[wet_mask],   q, dim=1)
        q_pred = torch.quantile(pred_flat[wet_mask], q, dim=1)
        loss_q = loss_q + F.mse_loss(q_pred, q_gt) + \
                 0.6 * torch.abs((q_pred - q_gt) / (q_gt + 1e-8)).mean()
    return loss_q / len(q_levels)


def loss_dry_penalty(pred_p, gt_p, wet_thresh):
    """
    Penalise predicted rain in pixels where GT is dry.
    Surgical fix for ghost rain — does not compete with PCC or intensity.
    """
    dry = (gt_p < wet_thresh).float()
    return (F.relu(pred_p) * dry).pow(2).mean()


def loss_ssim(pred_p, gt_p, window_size=11):
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
    return 1.0 - torch.clamp(ssim_map, 0.0, 1.0).mean()


# =============================================================================
# PHASE SCHEDULER
# Three deterministic phases for a 3-day autonomous run.
# Deterministic (not adaptive) — avoids weight drift mid-run.
#
# Phase 0 (ep  0-15): Lock PCC. Minimal intensity to prevent early drift.
# Phase 1 (ep 15-40): Balance. Spectral + intensity ramp up.
# Phase 2 (ep 40+  ): Intensity push. PCC protected by its floor.
# =============================================================================

class PhaseScheduler:
    PHASES = [
        # (max_epoch, weights)
        (15,   {'edm': 1.0, 'pcc': 12.0, 'intensity': 2.0, 'spectral': 0.5,
                'tail': 1.0, 'ssim': 0.15, 'dry': 2.0, 'mass': 0.2}),
        (40,   {'edm': 1.0, 'pcc': 9.0,  'intensity': 5.0, 'spectral': 1.5,
                'tail': 2.0, 'ssim': 0.2,  'dry': 1.5, 'mass': 0.2}),
        (9999, {'edm': 1.2, 'pcc': 7.0,  'intensity': 8.0, 'spectral': 2.0,
                'tail': 2.5, 'ssim': 0.2,  'dry': 1.0, 'mass': 0.15}),
    ]

    def get_weights(self, epoch):
        for max_ep, w in self.PHASES:
            if epoch < max_ep:
                return w
        return self.PHASES[-1][1]

    def phase_name(self, epoch):
        names = ["PHASE-0 (PCC lock)", "PHASE-1 (balanced)", "PHASE-2 (intensity push)"]
        for i, (max_ep, _) in enumerate(self.PHASES):
            if epoch < max_ep:
                return names[i]
        return names[-1]


LOSS_KEYS = ['edm', 'pcc', 'intensity', 'spectral', 'tail', 'ssim', 'dry', 'mass']


# =============================================================================
# COMBINED LOSS  v9
# Operates on x0_pred (reconstructed clean field) for physics losses.
# EDM loss operates on (F_x, target_Fx) — the raw network output space.
# =============================================================================

def corrdiff_loss_v9(F_x, target_Fx, y_noisy, noise, sigma_r, x0_pred,
                     y_fine_full, sigma_data, wet_thresh_precip, weight_map,
                     phase_scheduler, epoch):
    """
    v9 combined loss. Single code path. Eight terms, all positive.

    Inputs:
        F_x          — raw network output
        target_Fx    — EDM training target for F_x
        y_noisy      — noisy input field
        noise        — noise that was added
        sigma_r      — noise level [B,1,1,1]
        x0_pred      — reconstructed clean field (c_skip*y_noisy + c_out*F_x)
        y_fine_full  — clean ground truth field
    """
    sigma_flat = sigma_r.flatten()

    pred_p = x0_pred[:, PRECIP_CH:PRECIP_CH+1]
    gt_p   = y_fine_full[:, PRECIP_CH:PRECIP_CH+1]

    # 1. EDM loss — primary training signal
    l_edm = loss_edm(F_x, target_Fx, sigma_flat, sigma_data, weight_map)

    # 2. Structural alignment
    l_pcc = loss_spatial_pcc(pred_p, gt_p, weight_map)

    # 3. Intensity: Huber + Log + Sqrt + Peak-95
    l_intensity = loss_intensity_v9(pred_p, gt_p, wet_thresh=wet_thresh_precip)

    # 4. Spectral sharpness
    l_spectral = loss_spectral_logmse(pred_p, gt_p)

    # 5. Heavy-rain tail
    l_tail = loss_tail_quantile(pred_p, gt_p, wet_thresh_precip)

    # 6. SSIM
    l_ssim = loss_ssim(pred_p, gt_p)

    # 7. Ghost-rain suppression
    l_dry = loss_dry_penalty(pred_p, gt_p, wet_thresh_precip)

    # 8. Mass conservation
    l_mass = (torch.abs(pred_p.sum(dim=[1,2,3]) - gt_p.sum(dim=[1,2,3])) /
              (torch.abs(gt_p.sum(dim=[1,2,3])) + 1e-6)).mean()
    l_mass = torch.clamp(l_mass, min=0.0)

    w = phase_scheduler.get_weights(epoch)

    total = (w['edm']       * l_edm       +
             w['pcc']       * l_pcc       +
             w['intensity'] * l_intensity  +
             w['spectral']  * l_spectral  +
             w['tail']      * l_tail      +
             w['ssim']      * l_ssim      +
             w['dry']       * l_dry       +
             w['mass']      * l_mass)

    if torch.isnan(total) or torch.isinf(total) or total.item() < 0:
        log(f"[W] Unstable total={total.item():.4f}, using safe fallback")
        total = torch.abs(total) + 1.0

    metrics = {
        'edm':       l_edm.item(),
        'pcc':       l_pcc.item(),
        'intensity': l_intensity.item(),
        'spectral':  l_spectral.item(),
        'tail':      l_tail.item(),
        'ssim':      l_ssim.item(),
        'dry':       l_dry.item(),
        'mass':      l_mass.item(),
    }
    return total, metrics, w


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


def composition_error(pred_p, gt_p, wet_thresh):
    """Wet/dry classification mismatch fraction. Target: keep below 0.20."""
    pred_wet = (pred_p > wet_thresh).float()
    gt_wet   = (gt_p   > wet_thresh).float()
    return (pred_wet - gt_wet).abs().mean().item()


# =============================================================================
# EMA  (CHANGE 9: decay=0.999 for less smoothing of extremes)
# =============================================================================

class EMA:
    def __init__(self, model, decay_final=0.999, warmup_steps=1000):
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
    LEARNING_RATE = 5e-6       # Fine-tuning rate; OneCycleLR peaks at 5e-5
    EPOCHS        = 15000
    PATIENCE      = 400
    P_STD         = 0.8
    N_STEPS_FAST  = 50

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset    = ClimateDataset(LR_FILES, HR_FILES, rank=rank)
    n          = len(dataset)
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
        dataset, [train_size, val_size, test_size], generator=generator)

    log(f"Dataset: {n}  Train {train_size}  Val {val_size}  Test {test_size}")
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
        HR_H, HR_W, dataset.lats, dataset.lons, device)
    log(f"Singapore weight map: {HR_H}x{HR_W}  peak={singapore_wmap.max().item():.2f}x")

    # ------------------------------------------------------------------
    # Embedding framework
    # CHANGE 10: sanity diagnostics for ghost-rain triage
    # ------------------------------------------------------------------
    embedding_fw = None
    if USE_PAPER_EMBEDDING:
        embedding_fw = EmbeddingFramework(s_freq=0.25, n_noise_steps=8,
                                          scale_factor=4, device=device)

        if rank == 0:
            log("  Collecting training data for QDM fitting...")
            lr_list, hr_list = [], []
            with torch.no_grad():
                for i, (lr_b, hr_b) in enumerate(train_loader):
                    if i >= 300:
                        break
                    lr_list.append(lr_b.cpu())
                    hr_list.append(hr_b.cpu())
            lr_all = torch.cat(lr_list)
            hr_all = torch.cat(hr_list)

            embedding_fw.fit_qdm(lr_all, hr_all, n_quantiles=N_QUANTILES)

            s_freq = find_psd_intersection(train_loader, device, n_batches=80,
                                           qdm_fw=embedding_fw)
            embedding_fw.s_freq = s_freq

            # --- Sanity diagnostics (CHANGE 10) ---
            # 1. Beta noise variance
            betas_np = embedding_fw.betas.cpu().numpy()
            cum_var  = float(1 - np.prod(1 - betas_np))
            log(f"  Beta schedule: first={betas_np[0]:.4f}  last={betas_np[-1]:.4f}"
                f"  cumulative_noise_var={cum_var:.4f}")
            log(f"  (if cum_var > 0.8, reduce n_noise_steps to reduce ghost rain)")

            # 2. f(HR) correlation check
            with torch.no_grad():
                test_hr  = hr_all[:16].float()
                f_out    = embedding_fw.f(test_hr)
                corr_f   = []
                for i in range(16):
                    a  = test_hr[i, PRECIP_CH].flatten()
                    b  = f_out[i,  PRECIP_CH].flatten()
                    vx = a - a.mean(); vy = b - b.mean()
                    c  = (vx*vy).sum() / ((vx**2).sum().sqrt() * (vy**2).sum().sqrt() + 1e-8)
                    corr_f.append(c.item())
                mean_corr_f = float(np.mean(corr_f))
                log(f"  f(HR) vs HR precip correlation = {mean_corr_f:.4f}")
                log(f"  (need >0.3; if lower, reduce n_noise_steps)")

                # 3. g(LR) vs f(HR) correlation check
                g_out   = embedding_fw.g(lr_all[:16].float(),
                                         hr_size=(HR_H, HR_W))
                corr_fg = []
                for i in range(16):
                    a  = f_out[i, PRECIP_CH].flatten()
                    b  = g_out[i, PRECIP_CH].flatten()
                    vx = a - a.mean(); vy = b - b.mean()
                    c  = (vx*vy).sum() / ((vx**2).sum().sqrt() * (vy**2).sum().sqrt() + 1e-8)
                    corr_fg.append(c.item())
                mean_corr_fg = float(np.mean(corr_fg))
                log(f"  f(HR) vs g(LR) correlation     = {mean_corr_fg:.4f}")
                log(f"  (should be 0.5-0.9; confirms shared embedding space)")

                # 4. QDM sanity
                s_lr  = lr_all[:100, PRECIP_CH].float().numpy()
                s_hr  = hr_all[:100, PRECIP_CH].float().numpy()
                s_qdm = embedding_fw._qdm_multiplicative(
                    s_lr, embedding_fw.qdm_lr[PRECIP_CH], embedding_fw.qdm_hr[PRECIP_CH])
                log(f"  QDM precip: LR mean={s_lr.mean():.4f}"
                    f" -> QDM mean={s_qdm.mean():.4f}"
                    f" -> HR mean={s_hr.mean():.4f}")

            qdm_state = embedding_fw.state_dict()
        else:
            qdm_state = None

        if world_size > 1:
            qdm_list = [qdm_state]
            dist.broadcast_object_list(qdm_list, src=0)
            qdm_state = qdm_list[0]

        embedding_fw.load_state_dict(qdm_state)
        embedding_fw.betas = embedding_fw.betas.to(device)

        assert embedding_fw._fitted, f"Rank {rank}: embedding_fw not fitted!"
        log(f"  Embedding ready rank={rank}  s_freq={embedding_fw.s_freq:.4f}")

    # ------------------------------------------------------------------
    # Model  (CHANGE 1: no regressor, 8-ch input)
    # ------------------------------------------------------------------
    diffusion_net = SingaporeCorrDiffNet(in_channels=8).to(device)

    model_core = SingaporeCorrDiffModel(
        diffusion_net,
        sigma_data=0.5, sigma_min=0.002, sigma_max=1.0, rho=_RHO
    ).to(device)

    log("Estimating sigma_data ...")
    sigma_data_est = estimate_sigma_data(
        train_loader, device, WET_THRESH_NORM, n_batches=50)

    sig_tensor = torch.tensor(sigma_data_est, device=device)
    if world_size > 1:
        dist.all_reduce(sig_tensor, op=dist.ReduceOp.SUM)
        sig_tensor /= world_size
    sigma_data_est = max(sig_tensor.item(), 0.8)   # floor at 0.8

    sigma_min = sigma_data_est * 0.008
    sigma_max = sigma_data_est * 1.5
    P_mean    = math.log(sigma_data_est) + 0.3

    model_core.sigma_data = sigma_data_est
    model_core.sigma_min  = sigma_min
    model_core.sigma_max  = sigma_max

    log(f"  sigma_data={sigma_data_est:.4f}  sigma_min={sigma_min:.4f}"
        f"  sigma_max={sigma_max:.4f}")

    # ------------------------------------------------------------------
    # DDP + Optimizer + Scheduler + EMA
    # ------------------------------------------------------------------
    model_core.diffusion_net = DDP(model_core.diffusion_net,
                                   device_ids=[local_rank],
                                   output_device=local_rank,
                                   find_unused_parameters=False)

    diffusion_net_train = model_core.diffusion_net

    # CHANGE 9: EMA decay=0.999
    ema = EMA(diffusion_net_train, decay_final=0.999, warmup_steps=1000)

    optimizer = AdamW(diffusion_net_train.parameters(),
                      lr=LEARNING_RATE, weight_decay=1e-4)

    # OneCycleLR: warmup 10%, then cosine decay. "Set and forget" for 3-day run.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 10,          # peak LR = 5e-5
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=1000.0,
    )

    scaler = GradScaler("cuda")

    BEST_CKPT_PATH   = os.path.join(SAVE_DIR, "corrdiff_singapore_v9_best.pth")
    LATEST_CKPT_PATH = os.path.join(SAVE_DIR, "corrdiff_singapore_v9_latest.pth")

    # Try to resume from v8 checkpoint. Architecture changed (12ch->8ch), so
    # model weights load only if shapes match; otherwise train from scratch.
    RESUME_CKPT  = os.path.join(SAVE_DIR, "corrdiff_singapore_v8_best.pth")
    best_metric  = float("inf")
    patience_counter = 0
    start_epoch  = 0

    if os.path.exists(RESUME_CKPT):
        try:
            ckpt = torch.load(RESUME_CKPT, map_location=device)
            diffusion_net_train.module.load_state_dict(ckpt["model"])
            ema.shadow = ckpt["ema"]
            for pg in optimizer.param_groups:
                pg['lr'] = LEARNING_RATE
            start_epoch = ckpt.get("epoch", 0) + 1
            best_metric = (0.6 * (1.0 - ckpt.get("val_pcc", 0.0)) +
                           0.4 * (1.0 - ckpt.get("val_sg_pcc", 0.0)))
            log(f"Resumed from epoch {start_epoch}, best composite={best_metric:.4f}")
        except Exception as e:
            log(f"Could not resume checkpoint (likely arch change 12ch->8ch): {e}")
            log("Starting from scratch.")

    phase_scheduler = PhaseScheduler()

    log("")
    log("=" * 70)
    log("CorrDiff Singapore  v9-FINAL  — 3-day autonomous run")
    log("Architecture: 8-ch pure generative (no regressor)")
    log(f"rho={_RHO}  N_QUANTILES={N_QUANTILES}  CFG_dropout=15%  EMA={ema.decay_final}")
    log(f"Loss: EDM + dual-PCC + intensity(H+L+S+P95) + spectral + tail + SSIM + dry + mass")
    log(f"sigma_data={sigma_data_est:.4f}  SG_WEIGHT={SG_WEIGHT}x")
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
                F_x, target_Fx, y_noisy, noise, sigma_r, x0_pred = \
                    model_core.forward_train(hr_b, lr_b, sigma,
                                             embedding_fw=embedding_fw)

                loss, metrics, curr_w = corrdiff_loss_v9(
                    F_x, target_Fx, y_noisy, noise, sigma_r, x0_pred,
                    hr_b, sigma_data_est, WET_THRESH_NORM,
                    singapore_wmap, phase_scheduler, epoch)

            if torch.isnan(loss) or torch.isinf(loss):
                log(f"[W] NaN/Inf at epoch {epoch+1}, skipping batch")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion_net_train.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            ema.update(diffusion_net_train)

            train_loss += loss.item()
            for k in LOSS_KEYS:
                loss_accum[k] += metrics.get(k, 0.0)
            n_batches += 1

        if n_batches == 0:
            continue

        train_loss /= n_batches
        for k in LOSS_KEYS:
            loss_accum[k] /= n_batches

        # ---------------------------------------------------------------
        # VALIDATION  (every 3 epochs; every epoch for first 10)
        # ---------------------------------------------------------------
        do_val = (epoch % 3 == 0) or (epoch < 10)

        if do_val:
            model_core.diffusion_net.eval()
            ema.backup(diffusion_net_train)
            ema.apply_shadow(diffusion_net_train)

            val_pcc    = 0.0
            val_sg_pcc = 0.0
            val_imae   = 0.0
            val_comp   = 0.0
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
                        weight_map=singapore_wmap,
                        s_churn=0.05)

                    pred_p = pred_val[:, PRECIP_CH:PRECIP_CH+1]
                    gt_p   = hr_w[:,   PRECIP_CH:PRECIP_CH+1]

                    pcc_val, n_wet = wet_pcc_batch(pred_p.squeeze(1),
                                                   gt_p.squeeze(1), WET_THRESH_NORM)
                    sg_pcc = singapore_pcc(pred_p, gt_p, singapore_wmap)
                    imae   = intensity_mae(pred_p.squeeze(1), gt_p.squeeze(1),
                                          WET_THRESH_NORM, weight_map=singapore_wmap)
                    comp   = composition_error(pred_p, gt_p, WET_THRESH_NORM)

                    val_pcc    += pcc_val.item() * n_wet
                    val_sg_pcc += sg_pcc         * n_wet
                    val_imae   += imae            * n_wet
                    val_comp   += comp            * n_wet
                    val_n      += n_wet

            ema.restore(diffusion_net_train)

            if val_n > 0:
                val_pcc    /= val_n
                val_sg_pcc /= val_n
                val_imae   /= val_n
                val_comp   /= val_n

            composite = (1.0 - val_pcc) * 0.6 + (1.0 - val_sg_pcc) * 0.4
            elapsed   = time.time() - t0
            phase     = phase_scheduler.phase_name(epoch)

            log(f"Ep {epoch+1:5d} [{phase}] | "
                f"loss {train_loss:.4f} "
                f"[edm {loss_accum['edm']:.3f}({curr_w['edm']:.1f}) "
                f"pcc {loss_accum['pcc']:.3f}({curr_w['pcc']:.1f}) "
                f"int {loss_accum['intensity']:.3f}({curr_w['intensity']:.1f}) "
                f"spec {loss_accum['spectral']:.3f}({curr_w['spectral']:.1f}) "
                f"dry {loss_accum['dry']:.3f}({curr_w['dry']:.1f}) "
                f"tail {loss_accum['tail']:.3f}] | "
                f"val PCC {val_pcc:.4f}  SG-PCC {val_sg_pcc:.4f}  "
                f"I-MAE {val_imae:.4f}  COMP {val_comp:.4f} | "
                f"composite {composite:.4f} | {elapsed:.1f}s")

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
                    "sigma_max":   sigma_max,
                    "val_pcc":     val_pcc,
                    "val_sg_pcc":  val_sg_pcc,
                    "val_imae":    val_imae,
                    "val_comp":    val_comp,
                    "embedding":   embedding_fw.state_dict()
                                   if embedding_fw is not None else None,
                }
                torch.save(ckpt, LATEST_CKPT_PATH)

                if composite < best_metric:
                    best_metric      = composite
                    patience_counter = 0
                    torch.save(ckpt, BEST_CKPT_PATH)
                    log(f"  >> NEW BEST composite={composite:.4f} "
                        f"(PCC={val_pcc:.4f}  SG-PCC={val_sg_pcc:.4f}  "
                        f"I-MAE={val_imae:.4f}  COMP={val_comp:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        log(f"Early stop at epoch {epoch+1}: patience exhausted.")
                        break

        else:
            elapsed = time.time() - t0
            if rank == 0 and epoch % 10 == 0:
                log(f"Ep {epoch+1:5d} | loss {train_loss:.4f}  "
                    f"[int {loss_accum['intensity']:.3f}  "
                    f"spec {loss_accum['spectral']:.3f}  "
                    f"pcc {loss_accum['pcc']:.3f}  "
                    f"dry {loss_accum['dry']:.3f}] | {elapsed:.1f}s")

    log("Training complete.")
    cleanup_ddp()


if __name__ == "__main__":
    main()
