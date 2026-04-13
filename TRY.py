# -*- coding: utf-8 -*-
"""
CorrDiff Stage-2 — FINAL WORKING TRAINING SCRIPT
==================================================
LOSS STACK (precipitation-specific, no VGG needed):
  1. EDM loss         — base denoiser stability
  2. MS-SSIM loss     — spatial structure, rain cell shape, edges
  3. FFT spectral loss — high-frequency power (kills blurring/ghost bridges)
  4. Dry-mask loss    — HARD penalty for predicting rain on dry pixels
  5. Intensity CDF loss — forces correct rain intensity distribution

CRITICAL FIXES vs old broken script:
  [F1] DIFF_SD measured at startup, used EVERYWHERE (train + sample)
  [F2] sigma_max = 3*diff_sd (data-adaptive)
  [F3] coarse_qdm broadcast correct shape
  [F4] EDM weight clamp = 100 (not 1000)
  [F5] Loss weights balanced: no single term dominates
  [F6] Ghost bridge killer: dry mask loss with margin
  [F7] Intensity loss: CDF matching in log space
"""

import os
import math
import time
import traceback
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
from Network import UNet, CorrDiffRegressor

# ================================================================
# CONFIG
# ================================================================
RF_PATH     = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
ORO_PATH    = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"
REG_PATH    = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/CorrDiff/checkpoints/regressor/regressor.pth"
SAVE_PATH   = "checkpoints/diffusion/diffusion_best.pth"
LATEST_PATH = "checkpoints/diffusion/diffusion_latest.pth"
QDM_PATH    = "checkpoints/diffusion/qdm_tables.npz"

BATCH_SIZE = 32          # smaller = more gradient signal per wet sample
LR         = 5e-5        # higher LR works well with rich loss signal
EPOCHS     = 15000
P_MEAN     = -1.2
P_STD      = 1.2
PRECIP_CH  = 0
LOG_EVERY  = 5
GAMMA      = 2.2

# Loss weights — tuned for precipitation
W_EDM      = 1.0    # base diffusion loss
W_SSIM     = 4.0    # spatial structure (most important for PCC)
W_FFT      = 1.0    # spectral sharpness (kills blurring)
W_DRY      = 3.0    # ghost bridge killer (HARD — must be highest or equal)
W_INTENS   = 1.5    # intensity distribution matching

WET_THRESH_LIN  = 0.1    # mm/day threshold for "wet"
DRY_MARGIN      = 0.02   # predict > this on dry pixel = penalized
SIGMA_MEAS_N    = 20     # batches to measure diff_sd

# ================================================================
# DDP SETUP
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
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= ws
    return t

# ================================================================
# GAMMA TRANSFORM
# ================================================================
def apply_gamma(x, g=GAMMA):
    return torch.sign(x) * torch.pow(torch.abs(x) + 1e-8, 1.0 / g)

def invert_gamma(x, g=GAMMA):
    return torch.sign(x) * torch.pow(torch.abs(x) + 1e-8, g)

# ================================================================
# EDM PRECONDITIONING
# ================================================================
def edm_params(sigma, sd):
    s2, sd2 = sigma**2, sd**2
    return (
        sd2 / (s2 + sd2),                        # c_skip
        sigma * sd / (s2 + sd2).sqrt(),           # c_out
        1.0 / (s2 + sd2).sqrt(),                  # c_in
        0.25 * sigma.log()                        # c_noise
    )

def edm_loss(pred, target, sigma, sd):
    w = ((sigma**2 + sd**2) / (sigma * sd)**2).clamp(max=100.0)
    return (w.view(-1, 1, 1, 1) * (pred - target)**2).mean()

# ================================================================
# MS-SSIM LOSS  (spatial structure)
# ================================================================
class MSSSIM(nn.Module):
    """
    Multi-Scale SSIM — captures rain cell structure at multiple scales.
    Works on log1p precipitation maps.
    Better than plain SSIM: captures both fine rain cells and large systems.
    """
    def __init__(self, window_size=11, levels=3):
        super().__init__()
        self.levels = levels
        self.ws = window_size
        # Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)   # [ws, ws]
        self.register_buffer("window", window.view(1, 1, window_size, window_size))

    def _ssim(self, x, y):
        C1, C2 = 0.01**2, 0.03**2
        pad = self.ws // 2
        mu_x  = F.conv2d(x, self.window, padding=pad, groups=1)
        mu_y  = F.conv2d(y, self.window, padding=pad, groups=1)
        mu_xx = F.conv2d(x*x, self.window, padding=pad, groups=1) - mu_x**2
        mu_yy = F.conv2d(y*y, self.window, padding=pad, groups=1) - mu_y**2
        mu_xy = F.conv2d(x*y, self.window, padding=pad, groups=1) - mu_x*mu_y
        num = (2*mu_x*mu_y + C1) * (2*mu_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (mu_xx + mu_yy + C2)
        return (num / den.clamp(min=1e-8)).mean()

    def forward(self, pred, target):
        # Add a small epsilon to the max to avoid div by zero on dry days
        mx = target.flatten(1).max(1).values.view(-1, 1, 1, 1).clamp(min=1e-2)
        
        # Scale to [0, 1] but keep the relative magnitude
        p  = (pred   / mx).clamp(0, 1)
        t  = (target / mx).clamp(0, 1)
        
        loss = 0.0
        for _ in range(self.levels):
            # Clamp the SSIM result to [0, 1] before subtracting from 1.0
            ssim_val = self._ssim(p, t).clamp(0, 1)
            loss = loss + (1.0 - ssim_val)
            p = F.avg_pool2d(p, 2)
            t = F.avg_pool2d(t, 2)
        return loss / self.levels
# ================================================================
# FFT SPECTRAL LOSS  (kills blurring / ghost bridges)
# ================================================================
def fft_loss(pred, target):
    """
    Penalize mismatch in 2D power spectrum.
    Blurry predictions → missing high-freq power → large loss.
    Ghost bridges → spurious freq components → large loss.
    Works in log1p space directly.
    """
    P = torch.fft.rfft2(pred,   norm="ortho")
    T = torch.fft.rfft2(target, norm="ortho")
    # Power spectrum difference (log scale — matches human perception of rain textures)
    P_pow = torch.log1p(P.abs())
    T_pow = torch.log1p(T.abs())
    return F.l1_loss(P_pow, T_pow)

# ================================================================
# DRY MASK LOSS  (ghost bridge killer)
# ================================================================
def dry_mask_loss(pred_lin, target_lin, margin=DRY_MARGIN):
    dry_mask = (target_lin < WET_THRESH_LIN)
    if not dry_mask.any():
        return pred_lin.new_tensor(0.0)
    
    violation = (pred_lin[dry_mask] - margin).clamp(min=0.0)
    # Squaring the violation makes the model fear "light rain" on dry pixels even more
    return (violation**2).mean() * 10.0
# ================================================================
# INTENSITY CDF LOSS  (correct rain intensity distribution)
# ================================================================
def intensity_cdf_loss(pred_lin, target_lin, n_bins=64):
    """
    Match CDF of rain intensities between pred and target.
    Prevents model from predicting correct spatial structure
    but wrong intensity range (too weak or too strong).
    Only computed on wet pixels.
    """
    wet = target_lin.flatten(1).max(1).values > WET_THRESH_LIN
    if not wet.any():
        return pred_lin.new_tensor(0.0)

    p = pred_lin[wet].flatten()
    t = target_lin[wet].flatten()

    # Sort-based CDF matching (differentiable approximation)
    p_sorted = torch.sort(p)[0]
    t_sorted = torch.sort(t)[0]

    # Subsample to n_bins points for efficiency
    idx = torch.linspace(0, len(p_sorted)-1, n_bins, device=p.device).long()
    return F.l1_loss(p_sorted[idx], t_sorted[idx])

# ================================================================
# METRICS
# ================================================================
def batch_pcc(pred, target):
    p  = pred.flatten(1)
    t  = target.flatten(1)
    vp = p - p.mean(1, keepdim=True)
    vt = t - t.mean(1, keepdim=True)
    num = (vp * vt).sum(1)
    den = (vp.pow(2).sum(1).sqrt() * vt.pow(2).sum(1).sqrt()).clamp(min=1e-8)
    return (num / den).clamp(-1, 1).mean()

def wpcc(pred_lin, target_lin):
    wet = target_lin.flatten(1).max(1).values > WET_THRESH_LIN
    if not wet.any(): return pred_lin.new_tensor(0.0)
    return batch_pcc(pred_lin[wet], target_lin[wet])

# ================================================================
# EMA
# ================================================================
class EMA:
    def __init__(self, model, decay=0.9995, warmup=1000):
        self.decay   = decay
        self.warmup  = warmup
        self.step    = 0
        self.shadow  = {n: p.data.clone() for n, p in model.named_parameters()
                        if p.requires_grad}
        self._backup = None

    def _d(self):
        frac = min(self.step / max(self.warmup, 1), 1.0)
        return 0.99 + frac * (self.decay - 0.99)

    def update(self, model):
        self.step += 1
        d = self._d()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(d).add_(p.data, alpha=1-d)

    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

    def backup(self, model):
        self._backup = {n: p.data.clone() for n, p in model.named_parameters()
                        if p.requires_grad}

    def restore(self, model):
        if not self._backup: return
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = None

# ================================================================
# QDM
# ================================================================
class QDM:
    def __init__(self, n_q=1500):
        self.n_q      = n_q
        self.q_levels = np.linspace(0, 1, n_q)
        self.q_coarse = None
        self.q_obs    = None

    def load(self, path, rank=0, ws=1, dev=None):
        if is_main(rank): print(f"[QDM] Loading {path}")
        d  = np.load(path)
        qc = torch.from_numpy(d["q_coarse"].astype(np.float32))
        qo = torch.from_numpy(d["q_obs"   ].astype(np.float32))
        self.q_levels = d["q_levels"]
        if float(qc.max()) < 1.0:
            raise RuntimeError("QDM in log space — delete and refit in linear space.")
        if ws > 1 and dist.is_initialized():
            qc, qo = qc.to(dev), qo.to(dev)
            dist.broadcast(qc, 0); dist.broadcast(qo, 0)
            dist.barrier()
            qc, qo = qc.cpu(), qo.cpu()
        self.q_coarse = qc.numpy()
        self.q_obs    = qo.numpy()

    def apply(self, coarse_log1p):
        pl  = coarse_log1p[:, PRECIP_CH]
        pln = torch.expm1(pl.clamp(min=0)).detach().cpu().numpy()
        B, H, W = pln.shape
        mp = np.empty_like(pln)
        for i in range(H):
            for j in range(W):
                mp[:, i, j] = np.interp(
                    pln[:, i, j], self.q_coarse[:, i, j], self.q_obs[:, i, j])
        ml = torch.log1p(torch.from_numpy(
            np.clip(mp, 0, None).astype(np.float32))).to(coarse_log1p.device)
        return (ml - pl).unsqueeze(1)

# ================================================================
# MEASURE DIFF_SD  (must run before training)
# ================================================================
@torch.no_grad()
def measure_diff_sd(loader, reg, dev, sd_reg, n_batches=SIGMA_MEAS_N):
    reg.eval()
    stds = []
    for i, b in enumerate(loader):
        if i >= n_batches: break
        fine = b["fine"].to(dev)[:, PRECIP_CH:PRECIP_CH+1]
        topo = b["topo"].to(dev)
        tf   = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
        xi   = b["coarse_qdm"].to(dev)
        mu   = F.softplus(reg(xi, topo=topo, global_features=tf), beta=5)
        raw  = (fine - mu) / sd_reg
        stds.append(apply_gamma(raw).std().item())
    val = float(np.mean(stds))
    print(f"[DIFF_SD] Measured gamma-residual std = {val:.4f}")
    return max(val, 0.05)   # safety floor

# ================================================================
# SAMPLING  (Euler-Maruyama, Karras schedule)
# ================================================================
@torch.no_grad()
def sample(model, mu, topo, tf, diff_sd, n_steps=50):
    model.eval()
    m = model.module if hasattr(model, "module") else model

    sigma_max = 3.0 * diff_sd
    sigma_min = 0.002
    rho       = 7.0

    steps  = torch.arange(n_steps+1, device=mu.device, dtype=torch.float32) / n_steps
    sigmas = (sigma_max**(1/rho) + steps*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho

    x = torch.randn_like(mu) * sigmas[0]
    for i in range(n_steps):
        s, sn = sigmas[i], sigmas[i+1]
        cs, co, ci, cn = edm_params(s.expand(mu.shape[0]), diff_sd)
        ni       = torch.cat([ci.view(-1,1,1,1)*x, mu, topo], 1).clamp(-10, 10)
        denoised = cs.view(-1,1,1,1)*x + co.view(-1,1,1,1)*m(ni, cn, global_features=tf)
        x        = x + (sn - s) * ((x - denoised) / s)

    return (mu + invert_gamma(x) * diff_sd).clamp(-5, 9)

# ================================================================
# MAIN TRAINING
# ================================================================
def train():
    rank, ws, lr_, dev = setup_ddp()

    for p in [RF_PATH, ORO_PATH, REG_PATH, QDM_PATH]:
        assert os.path.exists(p), f"Missing: {p}"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # QDM
    qdm = QDM()
    qdm.load(QDM_PATH, rank, ws, dev)

    # Build coarse_qdm
    ds_raw = UpscaleDataset(RF_PATH, ORO_PATH, split=None, normalize=True, device="cpu")
    T = len(ds_raw)

    if is_main(rank):
        print(f"[DATA] Building coarse_qdm for {T} samples …")
        chunks = []
        for i in range(0, T, 256):
            c = ds_raw.coarse[i:i+256]
            chunks.append((c + qdm.apply(c)).cpu())
        coarse_qdm_tensor = torch.cat(chunks).contiguous()
        print(f"[DATA] coarse_qdm: {coarse_qdm_tensor.shape}")
    else:
        coarse_qdm_tensor = torch.zeros(T, *ds_raw.coarse[0:1].shape[1:])

    if ws > 1 and dist.is_initialized():
        coarse_qdm_tensor = coarse_qdm_tensor.to(dev)
        dist.broadcast(coarse_qdm_tensor, src=0)
        dist.barrier()
        coarse_qdm_tensor = coarse_qdm_tensor.cpu()

    ds  = UpscaleDataset(RF_PATH, ORO_PATH, split="train", normalize=True,
                         device="cpu", coarse_qdm=coarse_qdm_tensor)
    n   = len(ds)
    trn = int(0.70 * n)
    van = int(0.10 * n)
    trl = DataLoader(Subset(ds, range(0, trn)), BATCH_SIZE,
                     shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val = DataLoader(Subset(ds, range(trn, trn+van)), BATCH_SIZE,
                     shuffle=False, num_workers=4, pin_memory=True)
    if is_main(rank): print(f"[DATA] Train={trn}  Val={van}")

    # Frozen regressor
    reg = CorrDiffRegressor(in_channels=1, out_channels=1).to(dev)
    ck  = torch.load(REG_PATH, map_location=dev)
    reg.load_state_dict(ck.get("ema_state_dict", ck.get("model_state_dict")))
    reg.eval()
    for p in reg.parameters(): p.requires_grad_(False)
    sd_reg = float(ck.get("sigma_data", 0.210693))
    if is_main(rank): print(f"[REG] Frozen, sigma_data={sd_reg:.6f}")

    # Measure DIFF_SD on rank 0, broadcast
    if is_main(rank):
        diff_sd = measure_diff_sd(trl, reg, dev, sd_reg)
    else:
        diff_sd = 0.0
    t = torch.tensor(diff_sd, device=dev)
    if ws > 1 and dist.is_initialized():
        dist.broadcast(t, src=0); dist.barrier()
    diff_sd = t.item()
    if is_main(rank): print(f"[DIFF_SD] Using {diff_sd:.4f} everywhere.\n")

    # Loss modules
    msssim = MSSSIM(window_size=11, levels=3).to(dev)

    # UNet
    model = UNet(in_channels=3, out_channels=1, global_dim=2).to(dev)
    if ws > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_])
    raw_model = model.module if ws > 1 else model
    ema = EMA(raw_model)

    opt    = AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.999))
    scaler = GradScaler(device=dev.type, init_scale=256.0,
                        growth_interval=500, backoff_factor=0.5)
    sched  = CosineAnnealingWarmRestarts(opt, T_0=500, T_mult=2, eta_min=LR * 0.01)

    # Resume
    start_epoch = 0
    best_wPCC   = -1.0
    for rpath in [SAVE_PATH, LATEST_PATH]:
        if os.path.exists(rpath):
            if is_main(rank): print(f"[RESUME] {rpath}")
            ck2 = torch.load(rpath, map_location=dev)
            raw_model.load_state_dict(ck2["model_state_dict"])
            if "ema_state_dict" in ck2:
                ema.shadow = {k: v.clone().to(dev) for k, v in ck2["ema_state_dict"].items()}
            if "optimizer_state_dict" in ck2:
                opt.load_state_dict(ck2["optimizer_state_dict"])
            start_epoch = ck2.get("epoch", 0) + 1
            best_wPCC   = ck2.get("val_wpcc", -1.0)
            if is_main(rank): print(f"  epoch={start_epoch}, best={best_wPCC:.4f}")
            break

    if is_main(rank):
        np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] {np_/1e6:.2f}M params")
        hdr = f"{'Ep':>6} | {'EDM':>6} | {'SSIM':>6} | {'FFT':>6} | {'DRY':>6} | {'INT':>6} | {'wPCC_tr':>7} | {'wPCC_val':>9}"
        print(hdr)
        print("-" * len(hdr))

    # ======================== TRAINING LOOP ========================
    for ep in range(start_epoch, EPOCHS):
        model.train()
        t0 = time.time()
        sum_edm = sum_ssim = sum_fft = sum_dry = sum_int = sum_wpcc = 0.0
        n_batch = 0

        for bi, b in enumerate(trl):
            try:
                fine  = b["fine"].to(dev)
                topo  = b["topo"].to(dev)
                tf    = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
                xi    = b["coarse_qdm"].to(dev)

                with torch.no_grad():
                    mu = F.softplus(reg(xi, topo=topo, global_features=tf), beta=5)

                fp = fine[:, PRECIP_CH:PRECIP_CH+1]

                # Ground-truth gamma-compressed residual
                tr_raw    = (fp - mu) / sd_reg
                tr_target = apply_gamma(tr_raw)

                # Sample noise level
                sig = (torch.randn(fp.shape[0], device=dev) * P_STD + P_MEAN
                       ).exp().clamp(0.002, 3.0 * diff_sd)

                cs, co, ci, cn = edm_params(sig, diff_sd)
                nr = tr_target + torch.randn_like(tr_target) * sig.view(-1, 1, 1, 1)
                ni = torch.cat([ci.view(-1,1,1,1)*nr, mu, topo], 1).clamp(-10, 10)

                opt.zero_grad(set_to_none=True)

                with autocast(device_type=dev.type, enabled=True):
                    F_th     = model(ni, cn, global_features=tf)
                    pr       = cs.view(-1,1,1,1)*nr + co.view(-1,1,1,1)*F_th

                    # Reconstruct in log-space and linear-space
                    pf       = mu + invert_gamma(pr) * sd_reg
                    pf_lin   = torch.clamp(torch.expm1(pf.clamp(-5, 9)),  0, 1500.0)
                    fp_lin   = torch.clamp(torch.expm1(fp.clamp(-5, 9)),  0, 1500.0)

                    # ---- LOSS 1: EDM (base denoiser) ----
                    l_edm  = edm_loss(pr, tr_target, sig, diff_sd)

                    # ---- LOSS 2: MS-SSIM (spatial structure) ----
                    # Work in log1p space: better dynamic range for precip
                    l_ssim = msssim(
                        torch.log1p(pf_lin),
                        torch.log1p(fp_lin)
                    )

                    # ---- LOSS 3: FFT spectral (kills blurring) ----
                    l_fft  = fft_loss(
                        torch.log1p(pf_lin),
                        torch.log1p(fp_lin)
                    )

                    # ---- LOSS 4: Dry mask (ghost bridge killer) ----
                    l_dry  = dry_mask_loss(pf_lin, fp_lin)

                    # ---- LOSS 5: Intensity CDF matching ----
                    l_int  = intensity_cdf_loss(pf_lin, fp_lin)

                    # ---- TOTAL LOSS ----
                    loss = (W_EDM   * l_edm  +
                            W_SSIM  * l_ssim +
                            W_FFT   * l_fft  +
                            W_DRY   * l_dry  +
                            W_INTENS* l_int)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
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
                if is_main(rank):
                    print(f"\n[ERROR] ep={ep} bi={bi}")
                    traceback.print_exc()
                raise

        sched.step()
        ni = 1.0 / max(n_batch, 1)

        te  = allreduce_mean(torch.tensor(sum_edm  * ni, device=dev), ws).item()
        ts  = allreduce_mean(torch.tensor(sum_ssim * ni, device=dev), ws).item()
        tf_ = allreduce_mean(torch.tensor(sum_fft  * ni, device=dev), ws).item()
        td  = allreduce_mean(torch.tensor(sum_dry  * ni, device=dev), ws).item()
        ti  = allreduce_mean(torch.tensor(sum_int  * ni, device=dev), ws).item()
        tw  = allreduce_mean(torch.tensor(sum_wpcc * ni, device=dev), ws).item()

        # ======================== VALIDATION ========================
        ema.backup(raw_model)
        ema.apply(raw_model)
        model.eval()

        vs = torch.tensor(0., device=dev)
        vn = torch.tensor(0,  device=dev, dtype=torch.long)

        with torch.no_grad():
            for bi, b in enumerate(val):
                if bi >= 30: break
                fine = b["fine"].to(dev)
                topo = b["topo"].to(dev)
                tf2  = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
                xi   = b["coarse_qdm"].to(dev)
                mu_v = F.softplus(reg(xi, topo=topo, global_features=tf2), beta=5)
                pf_v = sample(model, mu_v, topo, tf2, diff_sd)
                pfl  = torch.clamp(torch.expm1(pf_v.clamp(-5, 9)), 0, 1500.0)
                fpl  = torch.clamp(torch.expm1(fine[:, 0:1].clamp(-5, 9)), 0, 1500.0)
                wet  = fpl.flatten(1).max(1).values > 0.1
                if wet.any():
                    vs += batch_pcc(pfl[wet], fpl[wet])
                    vn += 1

        if ws > 1 and dist.is_initialized():
            dist.barrier()
            dist.all_reduce(vs, op=dist.ReduceOp.SUM)
            dist.all_reduce(vn, op=dist.ReduceOp.SUM)
            dist.barrier()

        vw = (vs / vn.clamp(min=1)).item()
        ema.restore(raw_model)
        el = time.time() - t0

        if is_main(rank) and ep % LOG_EVERY == 0:
            print(f"{ep:>6} | {te:>6.3f} | {ts:>6.3f} | {tf_:>6.3f} | "
                  f"{td:>6.3f} | {ti:>6.3f} | {tw:>7.4f} | {vw:>9.4f}  [{el:.0f}s]")

        # ======================== CHECKPOINT ========================
        if is_main(rank):
            ckpt = {
                "epoch": ep,
                "model_state_dict": raw_model.state_dict(),
                "ema_state_dict":   {k: v.cpu() for k, v in ema.shadow.items()},
                "optimizer_state_dict": opt.state_dict(),
                "val_wpcc":  vw,
                "sigma_data": sd_reg,
                "diff_sd":    diff_sd,
                "gamma":      GAMMA,
            }
            torch.save(ckpt, LATEST_PATH)

            if vw > best_wPCC:
                best_wPCC = vw
                ema.backup(raw_model)
                ema.apply(raw_model)
                best = dict(ckpt)
                best["model_state_dict"] = raw_model.state_dict()
                best["ema_state_dict"]   = raw_model.state_dict()
                torch.save(best, SAVE_PATH)
                ema.restore(raw_model)
                print(f"  ★ NEW BEST wPCC={vw:.4f} → {SAVE_PATH}")

    if ws > 1 and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
