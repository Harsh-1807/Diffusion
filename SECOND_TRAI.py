# -*- coding: utf-8 -*-
"""
CorrDiff Stage-2 — TEMPORAL DIFFUSION VERSION
==============================================

Key design:
  - Stage 1 regressor stays FROZEN and NON-TEMPORAL (as trained)
  - Temporal context is injected into the DIFFUSION UNet only
  - Conditioning stack per sample:
        [noisy_residual | mu | topo | lr_t-1 | lr_t-2 | lr_t-3]
    i.e. we concatenate T=3 previous coarse frames as extra channels
  - UNet in_channels is bumped accordingly (3_noise+mu+topo + T*C_coarse)
  - Everything else (EDM schedule, losses, EMA, manifest) unchanged

Why this is correct:
  - Regressor (μ) already captures the mean field from current LR
  - Diffusion learns the RESIDUAL — temporal context helps it learn
    storm propagation patterns in the residual, not just static structure
  - No ConvLSTM needed in Stage 2 for basic temporal — simple channel-cat
    is sufficient and much more stable to train

Setting T_COND = 0 recovers the original non-temporal Stage 2 exactly.
"""

import os, json, math, time, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Dataset import UpscaleDataset
from Network import UNet, CorrDiffRegressor, TemporalCorrDiffRegressor

# ================================================================
# CONFIG
# ================================================================
RF_PATH     = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
ORO_PATH    = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"
REG_PATH    = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/SECOND/checkpoints/regressor/regressor.pth"
SAVE_PATH   = "checkpoints/diffusion/diffusion_best.pth"
LATEST_PATH = "checkpoints/diffusion/diffusion_latest.pth"
CKPT_DIR    = "checkpoints/diffusion/"
MANIFEST    = "checkpoints/diffusion/ensemble_manifest.json"

BATCH_SIZE = 16
LR         = 5e-5
EPOCHS     = 15000
P_MEAN     = -1.2
P_STD      = 1.2
PRECIP_CH  = 0
LOG_EVERY  = 5
GAMMA      = 2.2

# ── TEMPORAL CONFIG ──────────────────────────────────────────────
# Number of previous LR frames to concatenate as diffusion conditioning.
# T_COND = 0  →  original non-temporal Stage-2 (safe fallback)
# T_COND = 5  →  recommended (yesterday + 2 days ago)
T_COND     = 5          # how many previous coarse frames to condition on
C_COARSE   = 1          # channels per coarse frame (precip only)
# ─────────────────────────────────────────────────────────────────

W_EDM = 1.0
W_SSIM = 0.0
W_FFT = 0.0
W_DRY = 0.0
W_INTENS = 0.0

WET_THRESH_LIN  = 0.1
DRY_MARGIN      = 0.02
SIGMA_MEAS_N    = 20

CKPT_SAVE_EVERY = 100
MAX_CKPTS_KEEP  = 16

# ================================================================
# COMPUTE UNet in_channels BASED ON T_COND
# ================================================================
# Base channels: noisy_residual(1) + mu(1) + topo(1) = 3
# Each temporal frame adds C_COARSE channels
UNET_IN_CH = 3 + T_COND * C_COARSE


# ================================================================
# DDP
# ================================================================
def setup_ddp():
    rank = int(os.environ.get("RANK", 0))
    ws   = int(os.environ.get("WORLD_SIZE", 1))
    lr_  = int(os.environ.get("LOCAL_RANK", 0))
    if ws > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(lr_)
    dev = torch.device(f"cuda:{lr_}" if torch.cuda.is_available() else "cpu")
    return rank, ws, lr_, dev

def is_main(r): return r == 0

def allreduce_mean(t, ws):
    if ws > 1 and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM); t /= ws
    return t


# ================================================================
# TEMPORAL FRAME BUILDER
# ================================================================

def build_temporal_cond(ds, batch_indices, t_cond, c_coarse, fine_size, dev):
    """
    Build a temporal conditioning tensor by looking back t_cond steps
    in the dataset's coarse field.

    Returns: [B, t_cond * c_coarse, H_fine, W_fine]
             — upsampled to fine resolution so it can be cat'd with mu/topo

    Boundary padding: if idx - k < 0, repeat frame 0.
    """
    if t_cond == 0:
        return None

    coarse_all = ds.coarse   # [N, 1, H_c, W_c] on CPU
    B = len(batch_indices)

    frames = []
    for k in range(t_cond, 0, -1):                # t-k, ..., t-2, t-1
        frame_k = []
        for idx in batch_indices:
            src = max(0, int(idx) - k)
            frame_k.append(coarse_all[src])       # [1, H_c, W_c]
        frame_k = torch.stack(frame_k, dim=0).to(dev)   # [B, 1, H_c, W_c]

        # Upsample to fine resolution
        frame_k_up = F.interpolate(
            frame_k.float(),
            size=(fine_size, fine_size),
            mode="bilinear",
            align_corners=False,
        )                                          # [B, 1, H_f, W_f]
        frame_k_up = torch.log1p(frame_k_up.clamp(0))
        frames.append(frame_k_up)

    return torch.cat(frames, dim=1)               # [B, t_cond, H_f, W_f]


# ================================================================
# GAMMA
# ================================================================
def apply_gamma(x, g=GAMMA):
    return torch.sign(x) * torch.pow(torch.abs(x) + 1e-8, 1.0 / g)

def invert_gamma(x, g=GAMMA):
    return torch.sign(x) * torch.pow(torch.abs(x) + 1e-8, g)


# ================================================================
# EDM
# ================================================================
def edm_params(sigma, sd):
    s2, sd2 = sigma**2, sd**2
    return (
        sd2 / (s2 + sd2),
        sigma * sd / (s2 + sd2).sqrt(),
        1.0 / (s2 + sd2).sqrt(),
        0.25 * sigma.log()
    )

def edm_loss(pred, target, sigma, sd):
    w = ((sigma**2 + sd**2) / (sigma * sd)**2)
    w = w.clamp(min=0.1, max=10.0)
    return (w.view(-1, 1, 1, 1) * (pred - target)**2).mean()


# ================================================================
# MS-SSIM
# ================================================================
class MSSSIM(nn.Module):
    def __init__(self, window_size=11, levels=3):
        super().__init__()
        self.levels = levels; self.ws = window_size
        sigma  = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2)); g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        self.register_buffer("window", window.view(1, 1, window_size, window_size))

    def _ssim(self, x, y):
        C1, C2 = 0.01**2, 0.03**2
        pad = self.ws // 2
        mu_x  = F.conv2d(x, self.window, padding=pad)
        mu_y  = F.conv2d(y, self.window, padding=pad)
        mu_xx = F.conv2d(x*x, self.window, padding=pad) - mu_x**2
        mu_yy = F.conv2d(y*y, self.window, padding=pad) - mu_y**2
        mu_xy = F.conv2d(x*y, self.window, padding=pad) - mu_x*mu_y
        num = (2*mu_x*mu_y + C1) * (2*mu_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (mu_xx + mu_yy + C2)
        return (num / den.clamp(min=1e-8)).mean()

    def forward(self, pred, target):
        mx = target.flatten(1).max(1).values.view(-1, 1, 1, 1).clamp(min=1e-2)
        p  = (pred   / mx).clamp(0, 1)
        t  = (target / mx).clamp(0, 1)
        loss = 0.0
        for _ in range(self.levels):
            loss = loss + (1.0 - self._ssim(p, t).clamp(0, 1))
            p = F.avg_pool2d(p, 2); t = F.avg_pool2d(t, 2)
        return loss / self.levels


# ================================================================
# LOSSES
# ================================================================
def fft_loss(pred, target):
    P = torch.fft.rfft2(pred,   norm="ortho")
    T = torch.fft.rfft2(target, norm="ortho")
    return F.l1_loss(torch.log1p(P.abs()), torch.log1p(T.abs()))

def dry_mask_loss(pred_lin, target_lin, margin=DRY_MARGIN):
    dry = target_lin < WET_THRESH_LIN
    if not dry.any(): return pred_lin.new_tensor(0.0)
    violation = (pred_lin[dry] - margin).clamp(min=0.0)
    return (violation**2).mean() * 10.0

def intensity_cdf_loss(pred_lin, target_lin, n_bins=64):
    wet = target_lin.flatten(1).max(1).values > WET_THRESH_LIN
    if not wet.any(): return pred_lin.new_tensor(0.0)
    p = pred_lin[wet].flatten(); t = target_lin[wet].flatten()
    p_s = torch.sort(p)[0]; t_s = torch.sort(t)[0]
    idx = torch.linspace(0, len(p_s)-1, n_bins, device=p.device).long()
    return F.l1_loss(p_s[idx], t_s[idx])


# ================================================================
# METRICS
# ================================================================
def batch_pcc(pred, target):
    p = pred.flatten(1); t = target.flatten(1)
    vp = p - p.mean(1, keepdim=True); vt = t - t.mean(1, keepdim=True)
    num = (vp * vt).sum(1)
    den = (vp.pow(2).sum(1).sqrt() * vt.pow(2).sum(1).sqrt()).clamp(min=1e-8)
    return (num / den).clamp(-1, 1).mean()

def wpcc(pred_lin, target_lin):
    wet = target_lin.flatten(1).max(1).values > WET_THRESH_LIN
    if not wet.any(): return pred_lin.new_tensor(0.0)
    return batch_pcc(pred_lin[wet], target_lin[wet])

def dry_hit_rate(pred_lin, target_lin):
    dry = target_lin.flatten(1).max(1).values < WET_THRESH_LIN
    if not dry.any(): return pred_lin.new_tensor(1.0)
    return (pred_lin[dry].flatten(1).max(1).values < DRY_MARGIN).float().mean()

def heavy_rmse(pred_lin, target_lin):
    wet_vals = target_lin[target_lin > WET_THRESH_LIN]
    if wet_vals.numel() < 10: return pred_lin.new_tensor(0.0)
    thr   = torch.quantile(wet_vals, 0.90)
    heavy = target_lin.flatten(1).max(1).values > thr
    if not heavy.any(): return pred_lin.new_tensor(0.0)
    return F.mse_loss(pred_lin[heavy], target_lin[heavy]).sqrt()
    
@torch.no_grad()
def sample_members(model, reg, xi, topo, tf, diff_sd, ds, idx, K=8, n_steps=25):
    """
    Return K independent diffusion members.
    Shape: [K, B, 1, H, W]
    """
    model.eval()
    members = []
    for _ in range(K):
        s = sample(
            model=model, reg=reg, xi=xi, topo=topo, tf=tf,
            diff_sd=diff_sd, ds_ref=ds, idx_list=idx, n_steps=n_steps,
        )
        members.append(s)
    return torch.stack(members, dim=0)


# ================================================================
# EMA
# ================================================================
class EMA:
    def __init__(self, model, decay=0.9995, warmup=1000):
        self.decay = decay; self.warmup = warmup; self.step = 0
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self._backup = None

    def _d(self):
        return 0.99 + min(self.step / max(self.warmup, 1), 1.0) * (self.decay - 0.99)

    def update(self, model):
        self.step += 1; d = self._d()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad: self.shadow[n].mul_(d).add_(p.data, alpha=1-d)

    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow: p.data.copy_(self.shadow[n])

    def backup(self, model):
        self._backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def restore(self, model):
        if not self._backup: return
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._backup: p.data.copy_(self._backup[n])
        self._backup = None


# ================================================================
# MEASURE DIFF_SD
# ================================================================
@torch.no_grad()
def measure_diff_sd(loader, reg, dev, sd_reg, ds, n_batches=SIGMA_MEAS_N):
    """Measure the std of gamma-transformed residuals (no temporal in regressor)."""
    reg.eval()
    stds = []
    for i, b in enumerate(loader):
        if i >= n_batches: break
        fine = b["fine"].to(dev)[:, PRECIP_CH:PRECIP_CH+1]
        topo = b["topo"].to(dev)
        tf   = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
        xi   = b["coarse"].to(dev)

        # Regressor is always called WITHOUT temporal (Stage 1 was non-temporal)
        mu  = F.softplus(reg(xi, topo=topo, global_features=tf), beta=5)
        raw = (fine - mu) / sd_reg
        stds.append(apply_gamma(raw).std().item())

    val = float(np.mean(stds))
    print(f"[DIFF_SD] Measured gamma-residual std = {val:.4f}")
    return max(val, 0.05)


# ================================================================
# SAMPLING (with temporal conditioning)
# ================================================================
@torch.no_grad()
def sample(
    model, reg, xi, topo, tf, diff_sd,
    ds_ref=None, idx_list=None,
    n_steps=25, fine_size=128,
):
    model.eval()
    m = model.module if hasattr(model, "module") else model

    # Regressor: no temporal (frozen Stage-1 weights)
    mu = F.softplus(reg(xi, topo=topo, global_features=tf), beta=5)

    # Temporal conditioning for diffusion UNet
    temp_cond = None
    if T_COND > 0 and ds_ref is not None and idx_list is not None:
        temp_cond = build_temporal_cond(
            ds_ref, idx_list, T_COND, C_COARSE, fine_size, xi.device
        )

    sigma_max = 3.0 * diff_sd; sigma_min = 0.002; rho = 7.0
    steps  = torch.arange(n_steps+1, device=xi.device, dtype=torch.float32) / n_steps
    sigmas = (sigma_max**(1/rho) + steps*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho

    x = torch.randn_like(mu) * sigmas[0]

    def denoise(x_in, sig):
        sig_b          = sig.expand(mu.shape[0])
        cs, co, ci, cn = edm_params(sig_b, diff_sd)
        # Build input: [ci*x | mu | topo | temporal_cond]
        parts = [ci.view(-1,1,1,1) * x_in, mu, topo]
        if temp_cond is not None:
            parts.append(temp_cond)
        ni       = torch.cat(parts, dim=1).clamp(-10, 10)
        denoised = cs.view(-1,1,1,1)*x_in + co.view(-1,1,1,1)*m(ni, cn, global_features=tf)
        return denoised

    for i in range(n_steps):
        s, sn = sigmas[i], sigmas[i+1]
        Dx  = denoise(x, s)
        d   = (x - Dx) / s
        if i == n_steps - 1:
            x = Dx
        else:
            xn  = x + (sn - s) * d
            Dx2 = denoise(xn, sn)
            d2  = (xn - Dx2) / sn
            x   = x + (sn - s) * (d + d2) * 0.5

    return (mu + invert_gamma(x) * diff_sd).clamp(-5, 9)


# ================================================================
# MANIFEST
# ================================================================
def update_manifest(manifest_path, ckpt_path, wpcc_val, max_keep=MAX_CKPTS_KEEP):
    if os.path.exists(manifest_path):
        with open(manifest_path) as f: manifest = json.load(f)
    else:
        manifest = {"checkpoints": []}
    manifest["checkpoints"].append({"path": ckpt_path, "wpcc": float(wpcc_val)})
    manifest["checkpoints"].sort(key=lambda x: x["wpcc"], reverse=True)
    kept    = manifest["checkpoints"][:max_keep]
    removed = manifest["checkpoints"][max_keep:]
    for e in removed:
        p = e["path"]
        if os.path.exists(p) and p not in (SAVE_PATH, LATEST_PATH):
            try: os.remove(p)
            except Exception: pass
    manifest["checkpoints"] = kept
    with open(manifest_path, "w") as f: json.dump(manifest, f, indent=2)

@torch.no_grad()
def compute_sigma_data(loader, dev, n_batches=50):
    """Compute std of log1p fine precip over training data."""
    vals = []
    for i, b in enumerate(loader):
        if i >= n_batches: break
        fp = b["fine"].to(dev)[:, PRECIP_CH:PRECIP_CH+1]
        vals.append(fp.flatten())
    all_vals = torch.cat(vals)
    sd = all_vals.std().item()
    print(f"[SIGMA_DATA] Computed = {sd:.6f}")
    return max(sd, 0.01)
# ================================================================
# TRAIN
# ================================================================
def train():
    rank, ws, lr_, dev = setup_ddp()

    for p in [RF_PATH, ORO_PATH, REG_PATH]:
        assert os.path.exists(p), f"Missing: {p}"
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────
    ds  = UpscaleDataset(RF_PATH, ORO_PATH, split="train", normalize=True, device="cpu")
    n   = len(ds)
    trn = int(0.70 * n)
    van = int(0.10 * n)
    trl = DataLoader(Subset(ds, range(0, trn)), BATCH_SIZE,
                     shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val = DataLoader(Subset(ds, range(trn, trn+van)), BATCH_SIZE,
                     shuffle=False, num_workers=4, pin_memory=True)

    fine_size = ds.H

    if is_main(rank):
        print(f"[DATA] Train={trn}  Val={van}")
        print(f"[TEMPORAL] T_COND={T_COND}  →  UNet in_channels={UNET_IN_CH}")
        print(f"[FINE_SIZE] {fine_size}")

    # ── Load frozen Stage-1 regressor ────────────────────────────────
    ck = torch.load(REG_PATH, map_location=dev)
    use_temporal_reg = ck.get("use_temporal", False)

    reg_kwargs = dict(in_channels=1, out_channels=1, base_channels=96,
                      channel_mult=(1, 2, 4), num_blocks=2, global_dim=2)

    if use_temporal_reg:
        reg = TemporalCorrDiffRegressor(
            **reg_kwargs,
            seq_len=ck.get("seq_len", 5),
            lstm_layers=2,
            fine_size=fine_size,
        ).to(dev)
        if is_main(rank): print("[REG] TemporalCorrDiffRegressor (called without seq in Stage 2)")
    else:
        reg = CorrDiffRegressor(**reg_kwargs).to(dev)
        if is_main(rank): print("[REG] CorrDiffRegressor (non-temporal)")

    reg.load_state_dict(ck.get("ema_state_dict", ck.get("model_state_dict")))
    reg.eval()
    for p in reg.parameters():
        p.requires_grad_(False)

    # ── Compute sd_reg on rank 0, broadcast to all ranks ─────────────
    # sd_reg = std of the actual residual (fp - mu), not std of fp.
    # This correctly calibrates the EDM noise schedule.
    if is_main(rank):
        if "sigma_data" in ck:
            sd_reg = float(ck["sigma_data"])
            print(f"[REG] sigma_data loaded from checkpoint: {sd_reg:.6f}")
        else:
            print("[REG] WARNING: sigma_data not in checkpoint — computing residual std from data...")
            res_vals = []
            for i, b in enumerate(trl):
                if i >= 50: break
                fp_   = b["fine"].to(dev)[:, PRECIP_CH:PRECIP_CH+1]
                topo_ = b["topo"].to(dev)
                tf_   = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
                xi_   = b["coarse"].to(dev)
                with torch.no_grad():
                    mu_ = F.softplus(reg(xi_, topo=topo_, global_features=tf_), beta=5)
                res_vals.append((fp_ - mu_).flatten().cpu())
            sd_reg = max(torch.cat(res_vals).std().item(), 0.01)
            print(f"[REG] sigma_data (residual std) recomputed: {sd_reg:.6f}")
    else:
        sd_reg = 0.0

    sd_reg_t = torch.tensor(sd_reg, device=dev)
    if ws > 1 and dist.is_initialized():
        dist.broadcast(sd_reg_t, src=0)
    sd_reg = sd_reg_t.item()
    if is_main(rank): print(f"[REG] Using sigma_data={sd_reg:.6f}")

    # ── Measure diff_sd on rank 0, broadcast to all ranks ────────────
    if is_main(rank):
        diff_sd = measure_diff_sd(trl, reg, dev, sd_reg, ds)
    else:
        diff_sd = 0.0
    t_ = torch.tensor(diff_sd, device=dev)
    if ws > 1 and dist.is_initialized():
        dist.broadcast(t_, src=0)
        dist.barrier()
    diff_sd = t_.item()
    if is_main(rank): print(f"[DIFF_SD] Using {diff_sd:.4f}\n")

    # ── Models, optimizer, scheduler ─────────────────────────────────
    msssim = MSSSIM(window_size=11, levels=3).to(dev)

    model = UNet(
        in_channels=UNET_IN_CH,
        out_channels=1,
        global_dim=2,
    ).to(dev)

    if ws > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_])
    raw_model = model.module if ws > 1 else model
    ema = EMA(raw_model)

    opt    = AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.999))
    scaler = GradScaler(device=dev.type, init_scale=256.0,
                        growth_interval=500, backoff_factor=0.5)
    sched  = CosineAnnealingWarmRestarts(opt, T_0=500, T_mult=2, eta_min=LR * 0.01)

    start_epoch = 0
    best_wPCC   = -1.0

    # ── Resume from checkpoint if available ──────────────────────────
    for rpath in [SAVE_PATH, LATEST_PATH]:
        if os.path.exists(rpath):
            if is_main(rank): print(f"[RESUME] {rpath}")
            ck2 = torch.load(rpath, map_location=dev)
            saved_in_ch = ck2.get("unet_in_channels", UNET_IN_CH)
            if saved_in_ch != UNET_IN_CH:
                if is_main(rank):
                    print(f"  [WARN] Checkpoint in_channels={saved_in_ch} "
                          f"!= current {UNET_IN_CH}. Starting fresh.")
                break
            raw_model.load_state_dict(ck2["model_state_dict"])
            if "ema_state_dict" in ck2:
                ema.shadow = {k: v.clone().to(dev) for k, v in ck2["ema_state_dict"].items()}
            if "optimizer_state_dict" in ck2:
                opt.load_state_dict(ck2["optimizer_state_dict"])
            start_epoch = ck2.get("epoch", 0) + 1
            best_wPCC   = ck2.get("val_wpcc", -1.0)
            if is_main(rank): print(f"  epoch={start_epoch}  best={best_wPCC:.4f}")
            break

    if is_main(rank):
        np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] UNet {np_/1e6:.2f}M params  in_ch={UNET_IN_CH}\n")
        hdr = (f"{'Ep':>6} | {'EDM':>6} | {'SSIM':>6} | {'FFT':>6} | "
               f"{'DRY':>6} | {'INT':>6} | {'wPCC_tr':>7} | "
               f"{'wPCC_v':>7} | {'DHR':>5} | {'HRMSE':>6}")
        print(hdr)
        print("-" * len(hdr))

    # ================================================================
    # TRAINING LOOP
    # ================================================================
    for ep in range(start_epoch, EPOCHS):
        model.train()
        t0 = time.time()
        sum_edm = sum_ssim = sum_fft = sum_dry = sum_int = sum_wpcc = 0.0
        n_batch = 0

        for bi, b in enumerate(trl):
            try:
                fine = b["fine"].to(dev)
                topo = b["topo"].to(dev)
                tf   = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
                xi   = b["coarse"].to(dev)
                fp   = fine[:, PRECIP_CH:PRECIP_CH+1]

                # ── Temporal conditioning ─────────────────────────────
                temp_cond = None
                if T_COND > 0 and "idx" in b:
                    temp_cond = build_temporal_cond(
                        ds, b["idx"].tolist(), T_COND, C_COARSE, fine_size, dev
                    )
                elif T_COND > 0:
                    temp_cond = torch.zeros(
                        fp.shape[0], T_COND * C_COARSE,
                        fine_size, fine_size, device=dev
                    )

                # ── Frozen regressor forward ──────────────────────────
                with torch.no_grad():
                    mu = F.softplus(reg(xi, topo=topo, global_features=tf), beta=5)

                # ── EDM noise schedule ────────────────────────────────
                tr_raw    = (fp - mu) / sd_reg
                tr_target = apply_gamma(tr_raw)
                sig       = (torch.randn(fp.shape[0], device=dev) * P_STD + P_MEAN
                             ).exp()
                sig = sig.clamp(0.01, 1.5 * diff_sd)
                cs, co, ci, cn = edm_params(sig, diff_sd)
                nr = tr_target + torch.randn_like(tr_target) * sig.view(-1, 1, 1, 1)

                # ── Build UNet input ──────────────────────────────────
                parts = [ci.view(-1,1,1,1)*nr, mu, topo]
                if temp_cond is not None:
                    parts.append(temp_cond)
                ni = torch.cat(parts, dim=1).clamp(-10, 10)

                opt.zero_grad(set_to_none=True)
                with autocast(device_type=dev.type, enabled=True):
                    F_th   = model(ni, cn, global_features=tf)
                    pr     = cs.view(-1,1,1,1)*nr + co.view(-1,1,1,1)*F_th
                    pf     = mu + invert_gamma(pr) * sd_reg
                    pf_lin = torch.clamp(torch.expm1(pf.clamp(-5, 9)), 0, 1500.0)
                    fp_lin = torch.clamp(torch.expm1(fp.clamp(-5, 9)), 0, 1500.0)

                    l_edm  = edm_loss(pr, tr_target, sig, diff_sd)
                    l_ssim = msssim(torch.log1p(pf_lin), torch.log1p(fp_lin))
                    l_fft  = fft_loss(torch.log1p(pf_lin), torch.log1p(fp_lin))
                    l_dry  = dry_mask_loss(pf_lin, fp_lin)
                    l_int  = intensity_cdf_loss(pf_lin, fp_lin)

                    loss = l_edm + 0.05 * (1 - wpcc(pf_lin, fp_lin))

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                scaler.step(opt)
                scaler.update()
                ema.update(raw_model)

                with torch.no_grad():
                    sum_edm  += l_edm.item()
                    sum_ssim += l_ssim.item()
                    sum_fft  += l_fft.item()
                    sum_dry  += l_dry.item()
                    sum_int  += l_int.item()
                    sum_wpcc += wpcc(pf_lin, fp_lin).item()
                    n_batch  += 1

            except Exception:
                if is_main(rank): print(f"\n[ERROR] ep={ep} bi={bi}")
                traceback.print_exc()
                raise

        sched.step()
        ni  = 1.0 / max(n_batch, 1)
        te  = allreduce_mean(torch.tensor(sum_edm  * ni, device=dev), ws).item()
        ts  = allreduce_mean(torch.tensor(sum_ssim * ni, device=dev), ws).item()
        tf_ = allreduce_mean(torch.tensor(sum_fft  * ni, device=dev), ws).item()
        td  = allreduce_mean(torch.tensor(sum_dry  * ni, device=dev), ws).item()
        ti  = allreduce_mean(torch.tensor(sum_int  * ni, device=dev), ws).item()
        tw  = allreduce_mean(torch.tensor(sum_wpcc * ni, device=dev), ws).item()

        # ── VALIDATION ───────────────────────────────────────────────
        # ── VALIDATION ───────────────────────────────────────────────
        ema.backup(raw_model)
        ema.apply(raw_model)
        model.eval()
        
        vs = torch.tensor(0., device=dev)
        vd = torch.tensor(0., device=dev)
        vh = torch.tensor(0., device=dev)
        vn = torch.tensor(0,  device=dev, dtype=torch.long)
        
        DO_VAL = (ep % 5 == 0)
        
        with torch.no_grad():
            if DO_VAL:
                for bi, b in enumerate(val):
                    if bi >= 10: break   # ← faster
        
                    fine_v = b["fine"].to(dev)
                    topo_v = b["topo"].to(dev)
                    tf2    = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
                    xi_v   = b["coarse"].to(dev)
                    idx_v  = b["idx"].tolist() if "idx" in b else None
        
                    fpl = torch.clamp(
                        torch.expm1(fine_v[:, 0:1].clamp(-5, 9)),
                        0, 1500.0
                    )
        
                    # FAST ensemble (training-time)
                    members_v = sample_members(
                        model, reg,
                        xi_v, topo_v, tf2,
                        diff_sd=diff_sd,
                        ds=ds,
                        idx=idx_v,
                        K=4,           # ← reduced
                        n_steps=20,    # ← reduced
                    )
        
                    members_v_lin = torch.clamp(
                        torch.expm1(members_v.clamp(-5, 9)),
                        0, 1500.0
                    )
        
                    pfl_mean = members_v_lin.mean(0)
        
                    vs += wpcc(pfl_mean, fpl)
                    vd += dry_hit_rate(pfl_mean, fpl)
                    vh += heavy_rmse(pfl_mean, fpl)
                    vn += 1
        
        # ── DDP sync ───────────────────────────────────────────────
        if ws > 1 and dist.is_initialized():
            dist.barrier()
            for t__ in [vs, vd, vh, vn]:
                dist.all_reduce(t__, op=dist.ReduceOp.SUM)
            dist.barrier()
        
        # ── SAFE METRIC COMPUTATION ─────────────────────────────────
        if DO_VAL and vn > 0:
            vw   = (vs / vn).item()
            vdhr = (vd / vn).item()
            vhrm = (vh / vn).item()
        else:
            vw   = best_wPCC   # ← CRITICAL FIX
            vdhr = 0.0
            vhrm = 0.0
        
        ema.restore(raw_model)
        el = time.time() - t0
        
        # ── LOG ────────────────────────────────────────────────────
        if is_main(rank) and ep % LOG_EVERY == 0:
            print(f"{ep:>6} | {te:>6.3f} | {ts:>6.3f} | {tf_:>6.3f} | "
                  f"{td:>6.3f} | {ti:>6.3f} | {tw:>7.4f} | {vw:>7.4f} | "
                  f"{vdhr:>5.3f} | {vhrm:>6.3f}  [{el:.0f}s]")
        
        # ── CHECKPOINTING ───────────────────────────────────────────
        if is_main(rank):
            ckpt = {
                "epoch":                ep,
                "model_state_dict":     raw_model.state_dict(),
                "ema_state_dict":       {k: v.cpu() for k, v in ema.shadow.items()},
                "optimizer_state_dict": opt.state_dict(),
                "val_wpcc":             vw,
                "sigma_data":           sd_reg,
                "diff_sd":              diff_sd,
                "gamma":                GAMMA,
                "unet_in_channels":     UNET_IN_CH,
                "t_cond":               T_COND,
            }
        
            torch.save(ckpt, LATEST_PATH)
        
            # only update best when real validation happened
            if DO_VAL and vw > best_wPCC:
                best_wPCC = vw
                ema.backup(raw_model)
                ema.apply(raw_model)
        
                best = dict(ckpt)
                best["model_state_dict"] = raw_model.state_dict()
                best["ema_state_dict"]   = raw_model.state_dict()
        
                torch.save(best, SAVE_PATH)
                ema.restore(raw_model)
        
                print(f"  ★ NEW BEST wPCC={vw:.4f} → {SAVE_PATH}")
        
            # ensemble pool (only when validated)
            if DO_VAL and ep > 0 and ep % CKPT_SAVE_EVERY == 0 and vw > 0.0:
                ep_path = os.path.join(CKPT_DIR, f"diffusion_ep{ep:05d}.pth")
        
                ema.backup(raw_model)
                ema.apply(raw_model)
        
                ep_ckpt = dict(ckpt)
                ep_ckpt["model_state_dict"] = raw_model.state_dict()
                ep_ckpt["ema_state_dict"]   = raw_model.state_dict()
        
                torch.save(ep_ckpt, ep_path)
                ema.restore(raw_model)
        
                update_manifest(MANIFEST, ep_path, vw, MAX_CKPTS_KEEP)
        
        # ── CLEANUP ────────────────────────────────────────────────
        if ws > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    train()
