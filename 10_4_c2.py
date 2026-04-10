# -*- coding: utf-8 -*-
"""
CorrDiff Stage-2 Training - NUMERICALLY STABLE + PAPER-ALIGNED ACTIVE GAMMA
Fully compatible with the provided UpscaleDataset (log1p normalization)
FIX: Proper DDP checkpoint resume + synchronized validation
"""

import os
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import Subset
import math
from Dataset import UpscaleDataset
from Network import UNet, CorrDiffRegressor

# ====================== CONFIG ======================
RF_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
ORO_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"
REG_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/CorrDiff/checkpoints/regressor/regressor.pth"
SAVE_PATH = "checkpoints/diffusion/diffusion_final.pth"
LATEST_PATH = "checkpoints/diffusion/diffusion_latest.pth"
QDM_PATH = "checkpoints/diffusion/qdm_tables.npz"

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 15000
N_QUANTILES = 1500
P_MEAN = -1.2
P_STD = 1.8
WET_THRESH_LIN = 0.1
PRECIP_CH = 0
LOG_EVERY = 5

# ====================== DDP ======================
def setup_ddp():
    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    lr_ = int(os.environ.get("LOCAL_RANK", 0))
    if ws > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(lr_)
    dev = torch.device(f"cuda:{lr_}" if torch.cuda.is_available() else "cpu")
    return rank, ws, lr_, dev

def is_main(r):
    return r == 0

def red(t, ws):
    if ws > 1 and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= ws
    return t

# ====================== LEARNABLE GAMMA (ACTIVE) ======================
class LearnableGamma(torch.nn.Module):
    def __init__(self, init_gamma=2.2, min_gamma=0.5, max_gamma=3.0):
        super().__init__()
        self.log_gamma = torch.nn.Parameter(torch.tensor(math.log(init_gamma)))
        self.min_g = min_gamma
        self.max_g = max_gamma

    def _get_gamma(self):
        gamma_norm = torch.tanh(self.log_gamma)
        return self.min_g + (self.max_g - self.min_g) * (gamma_norm + 1) / 2

    def forward(self, x):
        """Apply paper Equation 3: sign(x) * |x|^(1/gamma)"""
        gamma = self._get_gamma()
        inv_gamma = 1.0 / (gamma + 1e-8)
        x_safe = torch.clamp(x, min=-100.0, max=100.0)
        result = torch.sign(x_safe) * torch.pow(torch.abs(x_safe) + 1e-8, inv_gamma)
        return torch.clamp(result, min=-1e4, max=1e4)

    def get_gamma(self):
        with torch.no_grad():
            return self._get_gamma().item()

# ====================== EMA ======================
class EMA:
    def __init__(self, model, decay_final=0.9995, warmup_steps=1000):
        self.decay_final = decay_final
        self.warmup_steps = warmup_steps
        self.step = 0
        self.shadow = {n: p.data.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.backup_params = None

    def _get_decay(self):
        frac = min(self.step / max(self.warmup_steps, 1), 1.0)
        return 0.99 + frac * (self.decay_final - 0.99)

    def update(self, model):
        self.step += 1
        decay = self._get_decay()
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1 - decay) * p.data + decay * self.shadow[n]

    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n])

    def backup(self, model):
        self.backup_params = {n: p.data.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    def restore(self, model):
        if self.backup_params is None:
            return
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup_params:
                p.data.copy_(self.backup_params[n])
        self.backup_params = None

# ====================== QDM ======================
class QDM:
    def __init__(self, n_q=N_QUANTILES):
        self.n_q = n_q
        self.q_levels = np.linspace(0, 1, n_q)
        self.q_coarse = None
        self.q_obs = None
        self._fitted = False

    def _sync(self, qc, qo, ws, dev):
        if ws > 1 and dist.is_initialized():
            qc = qc.to(dev)
            qo = qo.to(dev)
            dist.broadcast(qc, 0)
            dist.broadcast(qo, 0)
            dist.barrier()
            qc = qc.cpu()
            qo = qo.cpu()
        self.q_coarse = qc.numpy()
        self.q_obs = qo.numpy()
        self._fitted = True

    def load(self, path, rank=0, ws=1, dev=None):
        if is_main(rank):
            print(f"Loading QDM from {path}")
        d = np.load(path)
        qc = torch.from_numpy(d["q_coarse"].astype(np.float32))
        qo = torch.from_numpy(d["q_obs"].astype(np.float32))
        self.q_levels = d["q_levels"]
        mx = float(qc.max())
        if is_main(rank):
            print(f" Max coarse quantile: {mx:.4f} mm/day")
        if mx < 1.0:
            raise RuntimeError(f"QDM max={mx:.4f} looks like log space. Delete {path} and refit.")
        self._sync(qc, qo, ws, dev)

    def apply(self, coarse_log1p):
        pl = coarse_log1p[:, PRECIP_CH]
        pln = torch.expm1(pl.clamp(min=0)).detach().cpu().numpy()
        B, H, W = pln.shape
        mp = np.empty_like(pln)
        for i in range(H):
            for j in range(W):
                mp[:, i, j] = np.interp(pln[:, i, j], self.q_coarse[:, i, j], self.q_obs[:, i, j])
        ml = torch.log1p(torch.from_numpy(np.clip(mp, 0, None).astype(np.float32))).to(coarse_log1p.device)
        return (ml - pl).unsqueeze(1)

# ====================== GAMMA HELPER ======================
def apply_gamma(x, gamma):
    """Paper Equation 3"""
    return torch.sign(x) * torch.pow(torch.abs(x) + 1e-8, 1.0 / gamma)

# ====================== EDM ======================
def edm_params(sigma, sd):
    s = sigma.view(-1, 1, 1, 1)
    return (
        sd**2 / (s**2 + sd**2),
        s * sd / (s**2 + sd**2).sqrt(),
        1.0 / (s**2 + sd**2).sqrt(),
        0.25 * sigma.log()
    )

def edm_loss(pr, tr, sigma, sd):
    w = (sigma**2 + sd**2) / (sigma * sd)**2
    w = w.clamp(max=1000.0)
    return (w.view(-1, 1, 1, 1) * (pr - tr)**2).mean()

# ====================== METRICS ======================
def bpcc(p, t):
    if p.dim() == 1:
        vx = p - p.mean()
        vy = t - t.mean()
        return ((vx*vy).sum() / ((vx**2).sum().sqrt()*(vy**2).sum().sqrt()+1e-8)).clamp(-1, 1)
    if p.shape[-2:] != t.shape[-2:]:
        p = F.interpolate(p, size=t.shape[-2:], mode="bilinear", align_corners=False)
    p = p.flatten(1)
    t = t.flatten(1)
    vx = p - p.mean(1, keepdim=True)
    vy = t - t.mean(1, keepdim=True)
    return ((vx*vy).sum(1) / ((vx**2).sum(1).sqrt()*(vy**2).sum(1).sqrt()+1e-8)).clamp(-1, 1).mean()

def wpcc(p, t):
    wet = torch.expm1(t.flatten(1).clamp(0)).max(1).values > WET_THRESH_LIN
    return bpcc(p[wet], t[wet]) if wet.any() else p.new_tensor(0.)

def pcc_loss(pred, target):
    return 1 - bpcc(pred, target)

@torch.no_grad()
def sample_india(model, mu, topo, tf, sd, n_steps=40):
    model.eval()
    # EDM Standard Schedule (rho=7)
    steps = torch.arange(n_steps + 1, device=mu.device)
    t = steps / n_steps
    sigma_min, sigma_max, rho = 0.002, 80.0, 7.0
    sigmas = (sigma_max**(1/rho) + t*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    
    # Start with noise based on the max sigma
    x = torch.randn_like(mu) * sigmas[0]
    
    for i in range(n_steps):
        s_cur, s_next = sigmas[i], sigmas[i+1]
        
        # Precondition inputs for EDM UNet
        cs, co, ci, cn = edm_params(s_cur, sd)
        ni = torch.cat([ci * x, mu, topo], 1)
        ni = torch.clamp(ni, -10.0, 10.0)
        
        # Denoising step
        curr_model = model.module if hasattr(model, 'module') else model
        denoised = cs * x + co * curr_model(ni, cn.view(-1), global_features=tf)
        
        # Euler update step
        d = (x - denoised) / s_cur
        x = x + (s_next - s_cur) * d
        
    return mu + x * sd # Final residual added to regressor mean

# ====================== TRAINING LOOP ======================
def train():
    rank, ws, lr_, dev = setup_ddp()
    for path in [RF_PATH, ORO_PATH]:
        assert os.path.exists(path), f"Missing: {path}"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Load QDM
    qdm = QDM()
    if os.path.exists(QDM_PATH):
        qdm.load(QDM_PATH, rank, ws, dev)
    else:
        raise RuntimeError(f"QDM file not found at {QDM_PATH}. Run Stage-1 first.")

    # Dataset (uses the exact class you provided)
    ds_raw = UpscaleDataset(RF_PATH, ORO_PATH, split=None, normalize=True, device="cpu")

    if is_main(rank):
        coarse_qdm_list = []
        for i in range(0, len(ds_raw), 256):
            c = ds_raw.coarse[i:i+256]
            bias = qdm.apply(c)
            coarse_qdm_list.append((c + bias).cpu())
        coarse_qdm_tensor = torch.cat(coarse_qdm_list).contiguous()
        print(f" coarse_qdm computed: {coarse_qdm_tensor.shape}")
    else:
        T = len(ds_raw)
        coarse_qdm_tensor = torch.zeros(T, 1, ds_raw.coarse.shape[2], ds_raw.coarse.shape[3])

    if ws > 1 and dist.is_initialized():
        coarse_qdm_tensor = coarse_qdm_tensor.to(dev)
        dist.broadcast(coarse_qdm_tensor, src=0)
        dist.barrier()
        coarse_qdm_tensor = coarse_qdm_tensor.cpu()

    ds = UpscaleDataset(RF_PATH, ORO_PATH, split="train", normalize=True, device="cpu", coarse_qdm=coarse_qdm_tensor)

    n = len(ds)
    trn = int(0.70 * n)
    van = int(0.10 * n)
    trd = Subset(ds, range(0, trn))
    vad = Subset(ds, range(trn, trn + van))

    if is_main(rank):
        print(f"Split Train:{trn} Val:{van}")

    trl = DataLoader(trd, BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val = DataLoader(vad, BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

    # Regressor
    reg = CorrDiffRegressor(in_channels=1, out_channels=1).to(dev)
    ck = torch.load(REG_PATH, map_location=dev)
    reg.load_state_dict(ck.get("ema_state_dict", ck.get("model_state_dict")))
    reg.eval()
    for p in reg.parameters():
        p.requires_grad_(False)
    sd = float(ck.get("sigma_data", 0.210693))
    if is_main(rank):
        print(f"Regressor frozen. sigma_data={sd:.4f}\n")

    # Learnable Gamma (active)
    learn_gamma = LearnableGamma(init_gamma=2.2, min_gamma=0.5, max_gamma=10.0).to(dev)
    gamma_params = list(learn_gamma.parameters())

    # UNet
    model = UNet(in_channels=3, out_channels=1, global_dim=2).to(dev)
    if ws > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_])

    ema_model = model.module if ws > 1 else model
    ema = EMA(ema_model, decay_final=0.9995, warmup_steps=1000)

    opt = AdamW([
        {'params': model.parameters(), 'lr': LR, 'weight_decay': 1e-4},
        {'params': gamma_params, 'lr': LR * 1.0, 'weight_decay': 0.0}
    ])

    scaler = GradScaler(device=dev.type, init_scale=256.0, growth_interval=500, backoff_factor=0.5)

    # ====================== CHECKPOINT RESUME (FIX #1) ======================
    start_epoch = 0
    best = -1.0
    
    if os.path.exists(SAVE_PATH):
        if is_main(rank):
            print(f"[RESUME] Loading checkpoint from {SAVE_PATH}...")
        checkpoint = torch.load(SAVE_PATH, map_location=dev)
        
        # Load model state - handle DDP wrapper
        model_to_load = model.module if ws > 1 else model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        
        # Load EMA shadow if available
        if "ema_state_dict" in checkpoint:
            ema.shadow = {n: p.clone().detach().to(dev) for n, p in checkpoint["ema_state_dict"].items()}
            if is_main(rank):
                print(f"  EMA shadow restored")
        
        # Load optimizer state (optional - old checkpoints may not have it)
        if "optimizer_state_dict" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
            if is_main(rank):
                print(f"  Optimizer state restored")
        else:
            if is_main(rank):
                print(f"  WARNING: Old checkpoint - no optimizer state. Starting fresh optimizer.")
        
        # Load training state
        start_epoch = checkpoint.get("epoch", 0) + 1
        best = checkpoint.get("val_wpcc", -1.0)
        
        if is_main(rank):
            print(f"  Resumed at epoch {start_epoch}, best_wPCC={best:.4f}\n")

    if is_main(rank):
        np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"UNet: {np_/1e6:.2f}M params")
        print(f"{'Ep':>5} | {'Ledm':>7} | {'Lint':>7} | {'wPCC_tr':>7} | {'wPCC_val':>8} | γ   | s")
        print("-" * 65)

    # ====================== MAIN TRAINING LOOP ======================
    for ep in range(start_epoch, EPOCHS):
        model.train()
        t0 = time.time()
        ae = ai = aw = an = 0

        for bi, b in enumerate(trl):
            try:
                fine = b["fine"].to(dev)          # log-space
                coarse = b["coarse"].to(dev)
                topo = b["topo"].to(dev)
                tf = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
                xi = b["coarse_qdm"].to(dev)      # log-space

                with torch.no_grad():
                    mu = F.softplus(reg(xi, topo=topo, global_features=tf), beta=5)

                fp = fine[:, PRECIP_CH:PRECIP_CH+1]   # log-space

                # === ACTIVE GAMMA PIPELINE (Paper-aligned) ===
                tr_raw = (fp - mu) / sd                     # residual in log-space
                gamma_val = learn_gamma._get_gamma()
                tr_target = apply_gamma(tr_raw, gamma_val)  # gamma-compressed residual

                sig = (torch.randn(fine.size(0), device=dev) * P_STD + P_MEAN).exp().clamp(0.002, 80.0)
                nr = tr_target + torch.randn_like(tr_target) * sig.view(-1, 1, 1, 1)

                cs, co, ci, cn = edm_params(sig, sd)
                ni = torch.cat([ci * nr, mu, topo], 1)
                ni = torch.clamp(ni, -10.0, 10.0)

                opt.zero_grad(set_to_none=True)

                with autocast(device_type=dev.type, enabled=True):
                    Fth = model(ni, cn, global_features=tf)
                    pr = (cs * nr + co * Fth) * sd

                    alpha = 2.0 + (sig / sd).view(-1, 1, 1, 1)
                    pf = mu + alpha * pr                     # still in log-space

                    # Convert to linear space for loss & metrics
                    # Convert to linear space for loss & metrics
                    pf_lin = torch.expm1(torch.clamp(pf, min=-5.0, max=9.0))
                    fp_lin = torch.expm1(torch.clamp(fp, -5, 9))
                    # Increase max to 2000.0 to allow heavy tail events (>1000mm)
                    pf_lin = torch.clamp(pf_lin, min=0.0, max=1500.0)
                    fp_lin = torch.clamp(fp_lin, min=0.0, max=1500.0)

                    # Losses
                    le = edm_loss(pr, tr_target, sig, sd)

                    wet_mask = (fp_lin > WET_THRESH_LIN)
                    if wet_mask.any():
                        li = F.l1_loss(torch.log1p(pf_lin[wet_mask]), torch.log1p(fp_lin[wet_mask]))
                    else:
                        li = pf.new_tensor(0.0)

                    lpcc = pcc_loss(pf_lin, fp_lin)

                    dry_mask = ~wet_mask
                    if dry_mask.any():
                        dry_penalty = F.l1_loss(pf_lin[dry_mask], torch.zeros_like(pf_lin[dry_mask]))
                    else:
                        dry_penalty = pf.new_tensor(0.0)
                    
                    # Aggressive focus on spatial honesty (PCC weighted at 3.0)
                    loss = 0.3 * le + 1.0 * li + 2.0 * lpcc + 0.5 * dry_penalty

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(opt)
                scaler.update()

                ema.update(ema_model)

                with torch.no_grad():
                    ae += le.item()
                    ai += li.item()
                    pf_clean = mu + alpha * pr.detach()
                    aw += wpcc(pf_clean, fp).item()
                    an += 1

            except Exception:
                if is_main(rank):
                    print(f"\nERROR ep={ep} batch={bi}:")
                    traceback.print_exc()
                raise

        ni_ = 1. / max(an, 1)
        te = red(torch.tensor(ae * ni_, device=dev), ws).item()
        ti = red(torch.tensor(ai * ni_, device=dev), ws).item()
        tw = red(torch.tensor(aw * ni_, device=dev), ws).item()

        # ====================== VALIDATION (FIX #2: DDP SYNCHRONIZATION) ======================
        # Limit validation to exactly 3 batches to keep all ranks in sync
        val_batches_limit = 3
        
        ema.backup(ema_model)
        ema.apply_shadow(ema_model)
        model.eval()
        vs = torch.tensor(0., device=dev)
        vn = torch.tensor(0, device=dev, dtype=torch.long)

        with torch.no_grad():
            for bi, b in enumerate(val):
                # All ranks check the same condition
                if bi >= val_batches_limit:
                    break
                
                fine = b["fine"].to(dev)
                topo = b["topo"].to(dev)
                tf = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
                xi = b["coarse_qdm"].to(dev)
        
                mu = F.softplus(reg(xi, topo=topo, global_features=tf), beta=5)
                
                # USE THE SAMPLER with n_steps=50
                pf = sample_india(model, mu, topo, tf, sd, n_steps=50) 
        
                pf_lin = torch.expm1(pf.clamp(-5, 9))
                fp_lin = torch.expm1(fine[:, 0:1].clamp(-5, 9))
                
                wet = (fp_lin.flatten(1).max(1).values > WET_THRESH_LIN)
                if wet.any():
                    vs += bpcc(pf_lin[wet], fp_lin[wet])
                    vn += wet.sum()

        # Synchronization barrier - ensures all ranks finished validation
        if ws > 1 and dist.is_initialized():
            dist.barrier()
            dist.all_reduce(vs, op=dist.ReduceOp.SUM)
            dist.all_reduce(vn, op=dist.ReduceOp.SUM)
            dist.barrier()

        vw = (vs / vn.clamp(min=1)).item()
        ema.restore(ema_model)

        el = time.time() - t0

        if is_main(rank) and ep % LOG_EVERY == 0:
            gamma_val = learn_gamma.get_gamma()
            print(f"{ep:>5} | {te:>7.4f} | {ti:>7.4f} | {tw:>7.4f} | {vw:>8.4f} | {gamma_val:>5.2f} | {el:.0f}s")

        # ====================== CHECKPOINT SAVE (TWO-FILE SYSTEM) ======================
        if is_main(rank):
            raw_model = model.module if ws > 1 else model
            
            # 1. ALWAYS SAVE LATEST (Overwrites every epoch for crash recovery)
            torch.save({
                "epoch": ep,
                "model_state_dict": raw_model.state_dict(),
                "ema_state_dict": ema.shadow, # Save the actual shadow dict
                "optimizer_state_dict": opt.state_dict(),
                "val_wpcc": vw,
                "sigma_data": sd,
                "gamma": learn_gamma.get_gamma(),
            }, LATEST_PATH)

            # 2. SAVE BEST (Only if validation improved)
            if vw > best:
                best = vw
                ema.backup(raw_model)
                ema.apply_shadow(raw_model) # Bake EMA into the weights
                
                torch.save({
                    "epoch": ep,
                    "model_state_dict": raw_model.state_dict(), # This is now the EMA version
                    "ema_state_dict": raw_model.state_dict(),   # For compatibility
                    "optimizer_state_dict": opt.state_dict(),
                    "val_wpcc": vw,
                    "sigma_data": sd,
                    "gamma": learn_gamma.get_gamma(),
                }, SAVE_PATH)
                
                ema.restore(raw_model)
                print(f" -> [NEW BEST] wPCC_val={vw:.4f}. Saved to {SAVE_PATH}")

    if ws > 1 and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
