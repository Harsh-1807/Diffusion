# -*- coding: utf-8 -*-
"""Stage-2: Train CorrDiff Diffusion — GT-direct mode (sparse precip aware)
Flow Matching UNet predicts GT fine precip directly, not residual over mu.
mu used as conditioning input only (not added to output).
"""

import os, json, time, traceback
import torch, torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from Dataset import UpscaleDataset
from Network import UNet, CorrDiffRegressor, FlowMatching, QDM
import matplotlib.pyplot as plt
RF_PATH  = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
ORO_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"
D2M_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/era5_aligned_to_rf.nc"
REG_PATH = "checkpoints/regressor/regressor_best.pth"
CKPT_DIR = "checkpoints/diffusion/FM/"
SAVE     = "checkpoints/diffusion/FM/diffusion_best.pth"
LATEST   = "checkpoints/diffusion/FM/diffusion_latest.pth"
QDM_PATH = "checkpoints/diffusion/FM/qdm.pth"

BATCH         = 16
LR            = 1e-5
DROPOUT       = 0.1
EPOCHS        = 5000
PRECIP        = 0
T_COND        = 5
UNET_IN       = 2 + T_COND        # noisy(1) + mu(1) + tc(T_COND); d2m inside UNet
WET           = 0.1
CFG_DROP_PROB = 0.15
CFG_GUIDANCE  = 1.5
CKPT_EVERY    = 100
MAX_CKPT      = 16
QDM_BATCHES   = 50
QDM_N         = 2000
FM_STEPS      = 6

# ── Sparsity / GT-aware loss weights ────────────────────────────
W_FM     = 1.0    # Flow Matching MSE on GT
W_WET    = 1.25    # extra weight on wet cells
W_SPEC   = 0.3   # spectral
W_PCC    = 0.4   # PCC
W_FSS = 0.05  # Start with 0.5; increase if the texture is still too smooth


def compute_radially_averaged_psd(image_2d):
    f = np.fft.fft2(image_2d)
    fshift = np.fft.fftshift(f)
    psd2D = np.abs(fshift) ** 2
    y, x = np.indices(psd2D.shape)
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1]).astype(int)
    tbin = np.bincount(r.ravel(), psd2D.ravel())
    nr = np.bincount(r.ravel())
    return tbin / nr

def plot_corrdiff_psd(gt_data, pred_data, regressor_data, save_path):
    # Calculate PSD for each sample and average them
    psd_gt_list = [compute_radially_averaged_psd(gt_data[i]) for i in range(gt_data.shape[0])]
    psd_pred_list = [compute_radially_averaged_psd(pred_data[i]) for i in range(pred_data.shape[0])]
    psd_reg_list = [compute_radially_averaged_psd(regressor_data[i]) for i in range(regressor_data.shape[0])]
    
    psd_gt_mean = np.mean(psd_gt_list, axis=0)
    psd_pred_mean = np.mean(psd_pred_list, axis=0)
    psd_reg_mean = np.mean(psd_reg_list, axis=0)
    
    frequencies = np.arange(len(psd_gt_mean))
    
    plt.figure(figsize=(8, 6), facecolor='#111122')
    ax = plt.gca()
    ax.set_facecolor('#111122')
    
    plt.loglog(frequencies[1:], psd_gt_mean[1:], label='Ground Truth', color='#00ff00', linewidth=2.5, alpha=0.9)
    plt.loglog(frequencies[1:], psd_pred_mean[1:], label='CorrDiff (Stage 2)', color='#00bfff', linewidth=2.5, linestyle='--')
    plt.loglog(frequencies[1:], psd_reg_mean[1:], label='Regressor (Stage 1)', color='#ff3333', linewidth=2, linestyle=':')

    plt.title(f"Power Spectral Density", color='white', fontsize=14, pad=10)
    plt.xlabel("Spatial Wavenumber k", color='#aaaaaa', fontsize=12)
    plt.ylabel("Power Spectrum P(k)", color='#aaaaaa', fontsize=12)
    
    plt.grid(True, which="both", ls="-", alpha=0.2, color='white')
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values(): sp.set_edgecolor("#333355")
    
    legend = plt.legend(facecolor='#222233', edgecolor='#333355', fontsize=11)
    for text in legend.get_texts(): text.set_color("white")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
    plt.close()

# ── Sobel topo expansion ─────────────────────────────────────────

def compute_slope_aspect(elev: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32,
                       device=elev.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32,
                       device=elev.device).view(1,1,3,3)
    e  = elev.float()
    dx = F.conv2d(e, kx, padding=1)
    dy = F.conv2d(e, ky, padding=1)
    slope  = torch.sqrt(dx**2 + dy**2 + 1e-8)
    aspect = torch.atan2(dy, dx)
    def norm(t):
        mn = t.amin(); mx = t.amax()
        return 2*(t - mn)/(mx - mn + 1e-8) - 1
    return torch.cat([norm(e), norm(slope), norm(aspect)], dim=1)

def expand_topo(topo_1ch: torch.Tensor) -> torch.Tensor:
    return torch.cat([compute_slope_aspect(topo_1ch[i:i+1])
                      for i in range(topo_1ch.shape[0])], dim=0)


# ── DDP ──────────────────────────────────────────────────────────

def setup():
    rank = int(os.environ.get("RANK", 0))
    ws   = int(os.environ.get("WORLD_SIZE", 1))
    lr_  = int(os.environ.get("LOCAL_RANK", 0))
    if ws > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(lr_)
    return rank, ws, lr_, torch.device(f"cuda:{lr_}" if torch.cuda.is_available() else "cpu")

def ar(t, ws):
    if ws > 1 and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM); t /= ws
    return t


# ── Metrics ──────────────────────────────────────────────────────

def pcc(p, t):
    p = p.flatten(1); t = t.flatten(1)
    vp = p - p.mean(1, keepdim=True); vt = t - t.mean(1, keepdim=True)
    return (vp*vt).sum(1) / (vp.pow(2).sum(1).sqrt()*vt.pow(2).sum(1).sqrt()).clamp(1e-6)

def wpcc(p, o):
    wet = o.flatten(1).max(1).values > WET
    if wet.sum() < 5: return p.new_tensor(0.)
    return pcc(p[wet], o[wet]).clamp(-1, 1).mean()

def high_freq_spectral_loss(p, t, cutoff_k=10):
    Pm = torch.abs(torch.fft.rfft2(p.float())) + 1e-8
    Tm = torch.abs(torch.fft.rfft2(t.float())) + 1e-8
    
    # Create a mask that ONLY penalizes frequencies above cutoff_k
    _, _, h, w = Pm.shape
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=p.device), 
                          torch.linspace(0, 1, w, device=p.device), indexing='ij')
    radial_k = torch.sqrt(x**2 + y**2) * h # approximate wavenumber
    mask = (radial_k > cutoff_k).float()
    
    return F.mse_loss(torch.log1p(Pm) * mask, torch.log1p(Tm) * mask)

def dhr(p, o):
    dry = o.flatten(1).max(1).values < WET
    return (p[dry].flatten(1).max(1).values < WET).float().mean() if dry.any() else p.new_tensor(1.)

def hrmse(p, o):
    w = o[o > WET]
    if w.numel() < 10: return p.new_tensor(0.)
    thr = torch.quantile(w, .9)
    hv  = o.flatten(1).max(1).values > thr
    return F.mse_loss(p[hv], o[hv]).sqrt() if hv.any() else p.new_tensor(0.)
    
    
def crps(preds, obs):
    """
    preds: Ensemble predictions of shape [K, B, 1, H, W]
    obs: Ground truth of shape [B, 1, H, W]
    """
    K = preds.shape[0]
    
    # 1. Mean Absolute Error of each ensemble member to the observation
    mae = torch.abs(preds - obs).mean(0) 
    
    # 2. Pairwise ensemble spread (how diverse the predictions are)
    spread = torch.zeros_like(mae)
    for i in range(K):
        for j in range(K):
            spread += torch.abs(preds[i] - preds[j])
    spread = spread / (2 * K * K)
    
    # CRPS is MAE minus the spread
    return (mae - spread).mean()


def gt_aware_fm_loss(v_pred: torch.Tensor, v_target: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    wet_mask  = (x1 > torch.log1p(torch.tensor(WET, device=x1.device))).float()
    
    # Calculate P95 instead of P90 for a more aggressive peak focus
    p95       = torch.quantile(x1.flatten(1), 0.95, dim=1).view(-1,1,1,1)
    ext_mask  = (x1 > p95).float()

    # Apply a 3x boost to the top 5% of rainfall intensities
    weight = 1.0 + (W_WET - 1.0)*wet_mask + (1.1 * ext_mask) 
    return (weight * (v_pred - v_target)**2).mean()


# ── EMA ──────────────────────────────────────────────────────────

class EMA:
    def __init__(self, m, d=0.9995, wu=1000):
        self.d = d; self.wu = wu; self.step = 0
        self.s = {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
        self._b = None

    def _decay(self): return 0.99 + min(self.step/max(self.wu,1), 1.)*(self.d - 0.99)

    def update(self, m):
        self.step += 1; d = self._decay()
        with torch.no_grad():
            for n, p in m.named_parameters():
                if p.requires_grad: self.s[n].mul_(d).add_(p.data, alpha=1-d)

    def apply(self, m):
        for n, p in m.named_parameters():
            if p.requires_grad and n in self.s: p.data.copy_(self.s[n])

    def backup(self, m):
        self._b = {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}

    def restore(self, m):
        if not self._b: return
        for n, p in m.named_parameters():
            if p.requires_grad and n in self._b: p.data.copy_(self._b[n])
        self._b = None


# ── Temporal conditioning ─────────────────────────────────────────

def build_tc(ds, idx, H, W, dev):
    if not hasattr(ds, 'coarse') or idx is None:
        B = len(idx) if idx else 1
        return torch.zeros(B, T_COND, H, W, device=dev)
    frames = []
    for k in range(T_COND, 0, -1):
        fk = torch.stack([ds.coarse[max(0, int(i)-k)] for i in idx]).to(dev)
        fk = F.interpolate(fk.float(), (H, W), mode='bilinear', align_corners=False)
        frames.append(fk)
    return torch.cat(frames, 1)


# ── Flow Matching sampling (GT-direct, CFG) ───────────────────────

@torch.no_grad()
def sample_fm(model, reg, xi, topo_3ch, tf, ds, idx, d2m=None,
              n_steps=FM_STEPS, cfg_scale=CFG_GUIDANCE, qdm=None):
    """
    Sample GT fine precip directly.
    x starts from noise, ODE integrates to x1 = GT (log1p space).
    mu used as conditioning channel only — NOT added to output.
    """
    model.eval()
    m    = model.module if hasattr(model, "module") else model
    B    = xi.shape[0]
    H, W = topo_3ch.shape[2], topo_3ch.shape[3]

    mu = F.softplus(reg(xi, topo=topo_3ch, global_features=tf, d2m=d2m), beta=5)
    tc = build_tc(ds, idx, H, W, xi.device)

    x  = torch.randn_like(mu)
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_vec  = torch.full((B,), i * dt, device=x.device)
        x_cond = torch.cat([x, mu, tc], dim=1)            # [B, UNET_IN, H, W]

        v_c = m(x_cond, t_vec, topo=topo_3ch, global_features=tf, d2m=d2m)

        if cfg_scale > 1.0:
            x_unc    = torch.cat([x, torch.zeros_like(mu), torch.zeros_like(tc)], dim=1)
            cfg_mask = torch.ones(B, dtype=torch.bool, device=x.device)
            v_u = m(x_unc, t_vec, topo=topo_3ch, global_features=tf,
                    d2m=d2m, cfg_drop=cfg_mask)
            v = v_u + cfg_scale * (v_c - v_u)
        else:
            v = v_c

        x = x + dt * v

    # x is now GT in log1p space — decode directly
    pred_log = (mu + x).clamp(0, 5.0)
    phys     = torch.expm1(pred_log)

    if qdm is not None and qdm._fitted:
        phys = qdm.apply(phys)
        phys[phys < WET] = 0.

    return phys


@torch.no_grad()
def sample_K(model, reg, xi, tp3, tf, ds, idx, K=16, d2m=None, qdm=None):
    model.eval()
    # Returns [K, B, 1, H, W] instead of averaging
    return torch.stack([sample_fm(model, reg, xi, tp3, tf, ds, idx,
                                   d2m=d2m, qdm=qdm) for _ in range(K)])

@torch.no_grad()
def fit_qdm(model, reg, loader, ds, dev):
    model.eval(); ap = []; ao = []
    for bi, b in enumerate(loader):
        if bi >= QDM_BATCHES: break
        fv  = b["fine"].to(dev)
        tv3 = expand_topo(b["topo"].to(dev))
        tf  = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
        xi  = b["coarse"].to(dev)
        d2m = b["d2m"].to(dev) if "d2m" in b else None
        idx = b["idx"].tolist() if "idx" in b else None
        ap.append(sample_fm(model, reg, xi, tv3, tf, ds, idx, d2m=d2m).cpu())
        ao.append(torch.expm1(fv[:, :1].clamp(0, 5.0)).cpu())
    ap = torch.cat(ap).flatten(); ao = torch.cat(ao).flatten()
    wet = ao > WET
    print(f"  [QDM] wet={wet.float().mean():.3f} ({wet.sum():,})")
    if wet.sum() < 1000: return QDM(QDM_N)
    ap = ap[wet]; ao = ao[wet]
    if ap.numel() > 10_000_000:
        i = torch.randperm(ap.numel())[:10_000_000]; ap = ap[i]; ao = ao[i]
    q = QDM(QDM_N); q.fit(ap, ao); return q


def save_manifest(path, ckpt, wv, mk=MAX_CKPT):
    m = json.load(open(path)) if os.path.exists(path) else {"checkpoints": []}
    m["checkpoints"].append({"path": ckpt, "wpcc": float(wv)})
    m["checkpoints"].sort(key=lambda x: x["wpcc"], reverse=True)
    for e in m["checkpoints"][mk:]:
        p = e["path"]
        if os.path.exists(p) and p not in (SAVE, LATEST):
            try: os.remove(p)
            except: pass
    m["checkpoints"] = m["checkpoints"][:mk]
    json.dump(m, open(path, "w"), indent=2)

def fss_loss(pred, target, threshold=0.1, kernel_size=3):
    """
    Differentiable Fractions Skill Score (FSS) Loss.
    threshold: Rain intensity to consider a 'wet' pixel (e.g., 0.1mm)
    kernel_size: 3 for a 3x3 neighborhood.
    """
    # 1. Soft thresholding to create differentiable masks
    # We use a steep sigmoid (10.0) to approximate a binary rain mask
    p_mask = torch.sigmoid(10.0 * (pred - threshold)) 
    t_mask = (target > threshold).float() 
    
    # 2. Calculate local fractions (coverage) using average pooling
    # This turns a binary mask into a 'density' map of rain coverage
    p_frac = F.avg_pool2d(p_mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    t_frac = F.avg_pool2d(t_mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    # 3. FSS Formula: 1 - [ MSE(fractions) / (Mean(P^2) + Mean(T^2)) ]
    # We return (1 - FSS) as the loss to minimize.
    num = torch.mean((p_frac - t_frac)**2)
    den = torch.mean(p_frac**2) + torch.mean(t_frac**2) + 1e-8
    
    return num / den
# ── Training ─────────────────────────────────────────────────────

def train():
    rank, ws, lr_, dev = setup()
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 1. Load the full dataset
    ds = UpscaleDataset(RF_PATH, ORO_PATH, d2m_file=D2M_PATH,
                         split="train", normalize=True, device="cpu")
    
    # 2. Define your split indices (0.7 and 0.1)
    n = len(ds)
    trn_limit = int(0.70 * n)
    val_limit = trn_limit + int(0.10 * n)
    
    # 3. Create Subsets
    train_sub = Subset(ds, range(0, trn_limit))
    val_sub   = Subset(ds, range(trn_limit, val_limit))
    
    # 4. Initialize Distributed Samplers 
    # This is what ensures GPU 0-7 don't look at the same images!
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_sub, num_replicas=ws, rank=rank, shuffle=True
    ) if ws > 1 else None
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_sub, num_replicas=ws, rank=rank, shuffle=False
    ) if ws > 1 else None

    # 5. DataLoaders (Note: shuffle must be False when using a sampler)
    trl = DataLoader(train_sub, batch_size=BATCH, sampler=train_sampler,
                     shuffle=(train_sampler is None), # Only shuffle if not using sampler
                     num_workers=4, pin_memory=True, drop_last=True)
    
    val = DataLoader(val_sub, batch_size=BATCH, sampler=val_sampler,
                     num_workers=4, pin_memory=True)

    fs = ds.H
    if rank == 0: 
        print(f"[DATA] Total={n} Train={len(train_sub)} Val={len(val_sub)} Fine={fs} MODE=GT-direct")

    # --- REST OF YOUR SETUP ---

    # frozen regressor (mu = conditioning only)
    ck_r = torch.load(REG_PATH, map_location=dev)
    reg  = CorrDiffRegressor(
        in_channels=1, out_channels=1, base_channels=64,
        channel_mult=(1, 2, 4), num_blocks=2, global_dim=2,
        topo_channels=3, use_d2m=True
    ).to(dev)
    sd_key = "ema_state_dict" if "ema_state_dict" in ck_r else "model_state_dict"
    miss, unex = reg.load_state_dict(ck_r[sd_key], strict=False)
    if rank == 0: print(f"[REG] {len(miss)} missing {len(unex)} unexpected  (conditioning only)")
    reg.eval()
    for p in reg.parameters(): p.requires_grad_(False)

    fm = FlowMatching(n_steps=FM_STEPS, cfg_scale=CFG_GUIDANCE)

    model = UNet(
        in_channels=UNET_IN, out_channels=1, base_channels=96,
        channel_mult=(1, 2, 2, 4), num_res_blocks=2, dropout=DROPOUT, global_dim=2,
        use_bottleneck_attention=True, topo_channels=3,
        use_d2m=True, temporal_frames=T_COND
    ).to(dev)

    if ws > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_])
    raw = model.module if ws > 1 else model
    ema = EMA(raw)

    opt    = AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9, 0.999))
    scaler = GradScaler(device=dev.type, init_scale=256,
                        growth_interval=500, backoff_factor=0.5)
    sched  = CosineAnnealingWarmRestarts(opt, T_0=200, T_mult=2, eta_min=LR*0.01)

    start = 0; best = float('inf')
    qdm   = QDM(QDM_N)
    if os.path.exists(QDM_PATH):
        try: qdm = QDM.load(QDM_PATH); print("[QDM] Loaded")
        except: pass

    # Prioritize BEST over LATEST to escape the divergence zone
    for rp in [SAVE, LATEST]: 
        if os.path.exists(rp):
            ck2 = torch.load(rp, map_location=dev)
            raw.load_state_dict(ck2["model_state_dict"])
            if "ema_state_dict" in ck2:
                ema.s = {k: v.clone().to(dev) for k, v in ck2["ema_state_dict"].items()}
            
            start = ck2.get("epoch", 0) + 1
            
            # CRITICAL FIX: Look for 'val_crps'. 
            # If not found, use a high float (e.g., 10.0) so the first val step updates it.
            best = ck2.get("val_crps", 10.0) 
            
            if rank == 0: 
                print(f"[RESUME] Resumed from {rp} | ep={start} best_crps={best:.4f}")
            break

    if rank == 0:
        np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] UNet {np_/1e6:.2f}M  CFG={CFG_DROP_PROB}  GUIDE={CFG_GUIDANCE}")
        # ADDED FSS to print
        print(f"[LOSS]  FM={W_FM}  WET_BOOST={W_WET}x  SPEC={W_SPEC}  PCC={W_PCC}  FSS={W_FSS}")
        # FIXED HEADER to match the loop
        hdr = f"{'Ep':>6}|{'FM_L':>8}|{'wPCC_tr':>8}|{'CRPS_v':>8}|{'wPCC_v':>8}|{'HRMSE':>7}|QDM"
        print(hdr); print("-"*len(hdr))

    for ep in range(start, EPOCHS):
        if ws > 1:
            train_sampler.set_epoch(ep) # <--- ADD THIS
        model.train(); t0 = time.time(); se = sw = nb = 0.

        for b in trl:
            try:
                fp  = b["fine"].to(dev)[:, PRECIP:PRECIP+1]   # [B,1,H,W] log1p GT
                tp3 = expand_topo(b["topo"].to(dev))
                tf  = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
                xi  = b["coarse"].to(dev)
                d2m = b["d2m"].to(dev) if "d2m" in b else None
                idx = b["idx"].tolist() if "idx" in b else None

                with torch.no_grad():
                    mu = F.softplus(reg(xi, topo=tp3, global_features=tf, d2m=d2m), beta=5)

                tc = build_tc(ds, idx, fs, fs, dev)

                # FM on GT directly — x1 = fp, x0 = noise
                res_target = fp - mu
                x_t, t_fm, v_target = fm.get_train_sample(res_target)

                # model input: [noisy_GT(1), mu(1), tc(T_COND)]
                x_in     = torch.cat([x_t, mu, tc], dim=1)
                cfg_drop = torch.rand(fp.shape[0], device=dev) < CFG_DROP_PROB

                opt.zero_grad(set_to_none=True)
                with autocast(device_type=dev.type):
                    v_pred = model(x_in, t_fm, topo=tp3,
                                   global_features=tf, cfg_drop=cfg_drop, d2m=d2m)

                    l_fm = gt_aware_fm_loss(v_pred, v_target, fp)

                    # Physical decoding for auxiliary losses
                    t_exp    = t_fm.view(-1, 1, 1, 1)
                    x1_pred  = x_t + (1 - t_exp) * v_pred
                    pred_log = (mu + x1_pred).clamp(0, 5.0)
                    
                    pfl = torch.expm1(pred_log) # Predicted physical rain
                    fpl = torch.expm1(fp.clamp(0, 5.0)) # Truth physical rain

                    # --- NEW: FSS LOSS (3x3) ---
                    l_fss = fss_loss(pfl, fpl, threshold=WET, kernel_size=3)

                    l_spec = high_freq_spectral_loss(pfl, fpl,10)
                    pfl_pool = F.avg_pool2d(pfl, kernel_size=4, stride=4)
                    fpl_pool = F.avg_pool2d(fpl, kernel_size=4, stride=4)
                    l_pcc_macro = (1 - pcc(pfl_pool, fpl_pool)).mean()
                    
                    # Native PCC (Micro)
                    l_pcc_micro = (1 - pcc(pfl, fpl)).mean()
                    
                    # Combined Multi-scale PCC
                    l_pcc = 0.4 * l_pcc_macro + 0.6 * l_pcc_micro

                    # Combined Loss
                    loss = W_FM*l_fm + W_PCC*l_pcc + W_SPEC*l_spec + W_FSS*l_fss

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(opt); scaler.update(); ema.update(raw)

                with torch.no_grad():
                    se += l_fm.item()
                    sw += wpcc(pfl, fpl).item()
                    nb += 1
            except Exception:
                if rank == 0: traceback.print_exc(); raise

        sched.step()
        ni = 1. / max(nb, 1)
        te = ar(torch.tensor(se*ni, device=dev), ws).item()
        tw = ar(torch.tensor(sw*ni, device=dev), ws).item()

        ema.backup(raw); ema.apply(raw); model.eval()
        # Add vc for CRPS tracking
        vs = torch.tensor(0., device=dev); vd = torch.tensor(0., device=dev)
        vh = torch.tensor(0., device=dev); vc = torch.tensor(0., device=dev)
        vn = torch.tensor(0, device=dev, dtype=torch.long)
        DO_VAL = (ep % 5 == 0) or (ep == start)

        with torch.no_grad():
            if DO_VAL:
                for bi, b in enumerate(val):
                    if bi >= 15: break
                    fv    = b["fine"].to(dev)
                    tv3   = expand_topo(b["topo"].to(dev))
                    tf2   = torch.stack([b["doy"], b["hour"]], 1).float().to(dev)
                    xi_v  = b["coarse"].to(dev)
                    d2m_v = b["d2m"].to(dev) if "d2m" in b else None
                    idx_v = b["idx"].tolist() if "idx" in b else None
                    fpl   = torch.expm1(fv[:, :1].clamp(0, 5.0))
                    
                    if rank == 0 and bi == 0:
                        # 1. Generate exactly ONE prediction for the first validation batch
                        single_pred = sample_fm(model, reg, xi_v, tv3, tf2, ds, idx_v, d2m=d2m_v, qdm=qdm)
                        
                        # 2. Get the regressor's prediction (mu) in physical space
                        mu_val = F.softplus(reg(xi_v, topo=tv3, global_features=tf2, d2m=d2m_v), beta=5)
                        mu_phys = torch.expm1(mu_val.clamp(0, 5.0))
                        
                        # 3. Move everything to numpy arrays [Batch, Height, Width]
                        gt_np = fpl.cpu().numpy()[:, 0]
                        pred_np = single_pred.cpu().numpy()[:, 0]
                        reg_np = mu_phys.cpu().numpy()[:, 0]
                        
                        # 4. Save the plot
                        plot_path = os.path.join(CKPT_DIR, f"psd_ep{ep:05d}.png")
                        plot_corrdiff_psd(gt_np, pred_np, reg_np, save_path=plot_path)
                    
                    # 1. Get the full ensemble [K, B, 1, H, W]
                    ensemble = sample_K(model, reg, xi_v, tv3, tf2,
                                     ds, idx_v, K=32, d2m=d2m_v, qdm=qdm)
                    
                    # 2. Get the mean for legacy metrics
                    pfl = ensemble.mean(0)
                    
                    vs += wpcc(pfl, fpl); vd += dhr(pfl, fpl)
                    vh += hrmse(pfl, fpl)
                    # 3. Calculate CRPS on the raw ensemble
                    vc += crps(ensemble, fpl)
                    vn += 1

        if ws > 1 and dist.is_initialized():
            dist.barrier()
            for t__ in [vs, vd, vh, vc, vn]: dist.all_reduce(t__, op=dist.ReduceOp.SUM)
            dist.barrier()

        if DO_VAL and vn > 0:
            vw = (vs/vn).item(); vdhr = (vd/vn).item()
            vhrm = (vh/vn).item(); vcrps = (vc/vn).item()
        else:
            vw = 0.; vdhr = 0.; vhrm = 0.; vcrps = best

        qs = "✓" if qdm._fitted else "✗"
        ema.restore(raw); el = time.time() - t0

        if rank == 0 and ep % 5 == 0:
            # Update print statement to show CRPS
            if ep == start: 
                hdr = f"{'Ep':>6}|{'FM_L':>8}|{'wPCC_tr':>8}|{'CRPS_v':>8}|{'wPCC_v':>8}|{'HRMSE':>7}|QDM"
                print(hdr); print("-"*len(hdr))
            print(f"{ep:>6}|{te:>8.4f}|{tw:>8.4f}|{vcrps:>8.4f}|{vw:>8.4f}|{vhrm:>7.3f}|{qs} [{el:.0f}s]")

        if rank == 0:
            ck = {"epoch": ep,
                  "model_state_dict": raw.state_dict(),
                  "ema_state_dict": {k: v.cpu() for k, v in ema.s.items()},
                  "optimizer_state_dict": opt.state_dict(),
                  "val_crps": vcrps,
                  "unet_in": UNET_IN}
            torch.save(ck, LATEST)

            # Change saving logic: Save when CRPS is LOWER than best
            if DO_VAL and vcrps < best:
                best = vcrps; ema.backup(raw); ema.apply(raw)
                b2 = dict(ck); b2["model_state_dict"] = raw.state_dict()
                torch.save(b2, SAVE); ema.restore(raw)
                print(f"  ★ BEST CRPS={vcrps:.4f} (wPCC={vw:.4f})")
                print("  [QDM] Fitting..."); ema.backup(raw); ema.apply(raw)
                qdm = fit_qdm(model, reg, val, ds, dev)
                if qdm._fitted: qdm.save(QDM_PATH); print("  [QDM] Saved")
                ema.restore(raw)

            if DO_VAL and ep > 0 and ep % CKPT_EVERY == 0 and vw > 0:
                ep_p = os.path.join(CKPT_DIR, f"diffusion_ep{ep:05d}.pth")
                ema.backup(raw); ema.apply(raw)
                ec = dict(ck); ec["model_state_dict"] = raw.state_dict()
                torch.save(ec, ep_p); ema.restore(raw)
                save_manifest(os.path.join(CKPT_DIR, "manifest.json"), ep_p, vw)

    if ws > 1 and dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__": train()


# -*- coding: utf-8 -*-
"""
CorrDiff Network — v8  (2026 SOTA for small-data regional downscaling)
======================================================================
Changes from v7:
  * Flow Matching replaces DDPM  (straight trajectories, 4-8 Euler steps)
  * Temporal Self-Attention block in UNet bottleneck  [B*T,C,H,W] aware
  * Physics Guidance hooks at inference (non-negativity + mass conservation)
  * D2M + Topo-FiLM + CFG retained
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _g(ch, mx=32):
    for g in range(min(mx, ch), 0, -1):
        if ch % g == 0: return g
    return 1


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.p = nn.AdaptiveAvgPool2d(1)
        self.f = nn.Sequential(
            nn.Linear(ch, max(4, ch//r), bias=False), nn.ReLU(inplace=True),
            nn.Linear(max(4, ch//r), ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.f(self.p(x).view(b, c)).view(b, c, 1, 1)


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding)
    def forward(self, x):
        b, c, h, w = x.shape
        yg = torch.linspace(-1, 1, h, device=x.device).view(1,1,h,1).expand(b,1,h,w)
        xg = torch.linspace(-1, 1, w, device=x.device).view(1,1,1,w).expand(b,1,h,w)
        return self.conv(torch.cat([x, yg, xg], dim=1))


class FourierFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(channels, channels, 2) * 0.02)
        self.norm = nn.GroupNorm(_g(channels), channels)
        self.proj = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(self.norm(x))
        weight = torch.view_as_complex(self.complex_weight)
        out_fft = torch.einsum('bchw,cd->bdhw', x_fft, weight)
        return x + self.proj(torch.fft.irfft2(out_fft, s=(H, W)))


class ResConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_g(ic), ic), nn.SiLU(),
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.GroupNorm(_g(oc), oc), nn.SiLU(),
            nn.Conv2d(oc, oc, 3, padding=1))
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
        self.se   = SEBlock(oc)
    def forward(self, x): return self.skip(x) + self.se(self.net(x))


class BnAttn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        while ch % heads != 0 and heads > 1: heads -= 1
        self.h = heads; self.s = 8; self.scale = (ch//heads)**-0.5
        self.norm = nn.GroupNorm(_g(ch), ch)
        self.qkv  = nn.Conv2d(ch, ch*3, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        B, C, H, W = x.shape; s = min(self.s, H)
        lo = F.adaptive_avg_pool2d(self.norm(x), (s, s))
        qkv = self.qkv(lo).reshape(B, 3, self.h, C//self.h, s*s)
        q, k, v = qkv.unbind(1)
        a = torch.einsum('bhdn,bhdm->bhnm', q*self.scale, k).softmax(-1)
        o = torch.einsum('bhnm,bhdm->bhdn', a, v).reshape(B, C, s, s)
        return x + F.interpolate(self.proj(o), (H, W), mode='bilinear', align_corners=False)


# ================================================================
# Temporal Self-Attention  (bottleneck, lightweight token-per-frame)
# Input: [B*T, C, H, W]  — frames packed into batch dim
# ================================================================

class TemporalAttention(nn.Module):
    def __init__(self, ch, T=4, heads=4):
        super().__init__()
        while ch % heads != 0 and heads > 1: heads -= 1
        self.T = T; self.h = heads; self.scale = (ch//heads)**-0.5
        self.norm = nn.GroupNorm(_g(ch), ch)
        self.qkv  = nn.Linear(ch, ch*3, bias=False)
        self.proj = nn.Linear(ch, ch)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, T=None):
        T = T or self.T
        BT, C, H, W = x.shape
        if BT % T != 0: return x
        B = BT // T
        tok = F.adaptive_avg_pool2d(self.norm(x), 1).view(BT, C).view(B, T, C)
        qkv = self.qkv(tok).reshape(B, T, 3, self.h, C//self.h)
        q, k, v = qkv.unbind(2)
        a = torch.einsum('bthd,bshd->bths', q*self.scale, k).softmax(-1)
        out = torch.einsum('bths,bshd->bthd', a, v).reshape(B, T, C)
        out = self.proj(out).view(BT, C, 1, 1)
        return x + self.gate.tanh() * out


# ================================================================
# REGRESSOR  (Stage 1)
# ================================================================

class CorrDiffRegressor(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 channel_mult=(1,2,4), num_blocks=2, global_dim=2,
                 topo_channels=3, use_d2m=True, **kw):
        super().__init__()
        cms = list(channel_mult); st = base_channels; emb = base_channels*2
        self.use_d2m = use_d2m

        self.g_mlp = nn.Sequential(
            nn.Linear(global_dim, emb), nn.SiLU(), nn.Linear(emb, st)
        ) if global_dim > 0 else None

        self.r_stem = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            CoordConv2d(in_channels, st, 3, padding=1),
            nn.GroupNorm(_g(st), st), nn.SiLU())
        self.t_stem = nn.Sequential(
            CoordConv2d(topo_channels, st, 3, padding=1),
            nn.GroupNorm(_g(st), st), nn.SiLU())
        if use_d2m:
            self.d_stem = nn.Sequential(
                CoordConv2d(1, st, 3, padding=1),
                nn.GroupNorm(_g(st), st), nn.SiLU())

        self.r_enc = nn.ModuleList(); self.t_enc = nn.ModuleList()
        self.r_dn  = nn.ModuleList(); self.t_dn  = nn.ModuleList()
        self.sk_ch = []; rc = tc = st

        for li, m in enumerate(cms):
            oc = base_channels*m
            rb = nn.ModuleList(); tb = nn.ModuleList()
            for _ in range(num_blocks):
                rb.append(ResConv(rc, oc)); tb.append(ResConv(tc, oc)); rc = tc = oc
            self.r_enc.append(rb); self.t_enc.append(tb); self.sk_ch.append(rc+tc)
            last = li == len(cms)-1
            self.r_dn.append(nn.Identity() if last else nn.Conv2d(rc,rc,4,2,1))
            self.t_dn.append(nn.Identity() if last else nn.Conv2d(tc,tc,4,2,1))

        bn = base_channels*cms[-1]
        self.bn_proj = nn.Conv2d(rc+tc, bn, 1)
        self.bn_attn = BnAttn(bn, max(1, bn//64))
        self.bn_se   = SEBlock(bn)
        self.bn_mid  = ResConv(bn, bn)

        self.d_ups = nn.ModuleList(); self.d_blk = nn.ModuleList(); dc = bn
        for li, m in reversed(list(enumerate(cms))):
            oc = base_channels*m; sc = self.sk_ch[li]
            self.d_ups.append(nn.Identity() if li==len(cms)-1 else
                nn.Sequential(nn.ConvTranspose2d(dc,dc,4,2,1),
                               nn.GroupNorm(_g(dc),dc), nn.SiLU()))
            blks = nn.ModuleList(); ic2 = dc+sc
            for _ in range(num_blocks): blks.append(ResConv(ic2, oc)); ic2 = oc
            self.d_blk.append(blks); dc = oc

        self.out = nn.Sequential(
            nn.GroupNorm(_g(dc), dc), nn.SiLU(),
            nn.Conv2d(dc, out_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, topo, global_features=None, d2m=None):
        r = self.r_stem(x); t = self.t_stem(topo)
        if self.use_d2m and d2m is not None:
            d2m_up = F.interpolate(d2m, size=r.shape[-2:], mode='bilinear', align_corners=False)
            r = r + self.d_stem(d2m_up)
        if self.g_mlp is not None and global_features is not None:
            gs = self.g_mlp(global_features)[:,:,None,None]; r = r+gs; t = t+gs
        rs = []; ts = []
        for li in range(len(self.r_enc)):
            for rb, tb in zip(self.r_enc[li], self.t_enc[li]): r = rb(r); t = tb(t)
            rs.append(r); ts.append(t)
            r = self.r_dn[li](r); t = self.t_dn[li](t)
        
        f = self.bn_proj(torch.cat([r, t], 1))
        f = self.bn_attn(f); f = self.bn_se(f); f = self.bn_mid(f)
        d = f
        for li, (up, blks) in enumerate(zip(self.d_ups, self.d_blk)):
            lv = len(self.r_enc)-1-li; d = up(d)
            d = torch.cat([d, torch.cat([rs[lv], ts[lv]], 1)], 1)
            for b in blks: d = b(d)
        return self.out(d)


# ================================================================
# UNET  (Stage 2) — Flow Matching + Temporal Attn + CFG
# ================================================================

class ResBlock(nn.Module):
    def __init__(self, ic, oc, ec, down=False, up=False,
                 use_topo=True, topo_channels=3, dropout=0.1):
        super().__init__()
        self.rs = None
        if down: self.rs = nn.Conv2d(ic, ic, 4, 2, 1)
        if up:   self.rs = nn.ConvTranspose2d(ic, ic, 4, 2, 1)
        self.n1 = nn.GroupNorm(_g(ic), ic)
        self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.ep = nn.Linear(ec, oc)
        self.use_topo = use_topo
        if use_topo:
            self.topo_proj = nn.Conv2d(topo_channels, oc*2, kernel_size=3, padding=1)
            nn.init.zeros_(self.topo_proj.weight)
            nn.init.zeros_(self.topo_proj.bias)
        self.n2 = nn.GroupNorm(_g(oc), oc)
        self.dropout = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.se = SEBlock(oc)
        self.sk = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, e, topo=None):
        if self.rs: x = self.rs(x)
        h = self.c1(F.silu(self.n1(x))) + self.ep(F.silu(e))[:,:,None,None]
        h_norm = self.n2(h)
        if self.use_topo and topo is not None:
            t_res = F.interpolate(topo, size=h.shape[-2:], mode='bilinear', align_corners=False)
            gamma, beta = self.topo_proj(t_res).chunk(2, dim=1)
            h_norm = h_norm * (1 + gamma) + beta
            
        h_norm = self.dropout(h_norm)
        return self.se(self.c2(F.silu(h_norm))) + self.sk(x)


class UNet(nn.Module):
    """
    Flow-Matching UNet.
    Predicts velocity field v(x_t, t, cond).
    x_t = (1-t)*noise + t*data   |   v* = data - noise
    Loss: MSE(v_pred, v*)
    Sample: Euler  x_{t+dt} = x_t + dt*v   (6 steps default)
    """
    def __init__(self, in_channels, out_channels, base_channels=128,
                 channel_mult=(1,2,2,4), num_res_blocks=2, dropout=0.1, num_blocks=None,
                 global_dim=2, use_bottleneck_attention=True,
                 topo_channels=3, use_d2m=True, temporal_frames=3, **kw):
        super().__init__()
        nrb = num_blocks if num_blocks else num_res_blocks
        ec  = base_channels * 4
        self.topo_channels   = topo_channels
        self.use_d2m         = use_d2m
        self.temporal_frames = temporal_frames
        head_in = in_channels + (1 if use_d2m else 0)

        self.t_emb = nn.Sequential(
            nn.Linear(base_channels, ec), nn.SiLU(), nn.Linear(ec, ec))
        self.g_mlp = nn.Sequential(
            nn.Linear(global_dim, ec), nn.SiLU()) if global_dim else None

        self.head = CoordConv2d(head_in, base_channels, 3, padding=1)
        self.downs = nn.ModuleList(); self.ups = nn.ModuleList()
        ch = base_channels; sk = []

        for m in channel_mult:
            oc = base_channels * m
            for _ in range(nrb):
                self.downs.append(ResBlock(ch, oc, ec, topo_channels=topo_channels, dropout=dropout))
                ch = oc; sk.append(ch)
            self.downs.append(ResBlock(ch, ch, ec, down=True, topo_channels=topo_channels, dropout=dropout))
            sk.append(ch)

        self.m1 = ResBlock(ch, ch, ec, topo_channels=topo_channels, dropout=dropout)
        self.ma  = BnAttn(ch, max(1, ch//64)) if use_bottleneck_attention else nn.Identity()
        self.fft = FourierFilter(ch)
        #self.temp_attn = TemporalAttention(ch, T=temporal_frames) if temporal_frames > 1 else None
        self.m2 = ResBlock(ch, ch, ec, topo_channels=topo_channels, dropout=dropout)

        for m in reversed(channel_mult):
            oc = base_channels * m
            self.ups.append(ResBlock(ch+sk.pop(), oc, ec, up=True, topo_channels=topo_channels, dropout=dropout))
            ch = oc
            for _ in range(nrb):
                self.ups.append(ResBlock(ch+sk.pop(), oc, ec, topo_channels=topo_channels, dropout=dropout))
                ch = oc

        self.out = nn.Sequential(
            nn.GroupNorm(_g(ch), ch), nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1))

    def _temb(self, t):
        half = self.t_emb[0].in_features // 2
        freq = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000)/(half-1)))
        e = t.unsqueeze(1) * freq.unsqueeze(0) * 2 * math.pi
        return self.t_emb(torch.cat([e.sin(), e.cos()], -1))

    def forward(self, x, t, topo=None, global_features=None,
                cfg_drop=None, d2m=None, T=None):
        T = T or self.temporal_frames
        emb = self._temb(t)

        if global_features is not None and self.g_mlp:
            gf = global_features.clone()
            if cfg_drop is not None: gf[cfg_drop] = 0.
            emb = emb + self.g_mlp(gf)

        x_in    = x.clone()
        topo_in = topo.clone() if topo is not None else None

        if cfg_drop is not None and cfg_drop.any():
            x_in[cfg_drop, 1:] = 0.
            if topo_in is not None: topo_in[cfg_drop] = 0.

        if self.use_d2m and d2m is not None:
            d2m_in = d2m.clone()
            if cfg_drop is not None and cfg_drop.any():
                d2m_in[cfg_drop] = 0.
            d2m_res = F.interpolate(d2m_in, size=x_in.shape[-2:], mode='bilinear', align_corners=False)
            x_in = torch.cat([x_in, d2m_res], dim=1)

        h = self.head(x_in); sk = [h]
        for l in self.downs:   h = l(h, emb, topo_in); sk.append(h)
        h = self.m1(h, emb, topo_in)
        h = self.ma(h) if isinstance(self.ma, BnAttn) else h
        h = self.fft(h)
        #if self.temp_attn is not None: h = self.temp_attn(h, T=T)
        h = self.m2(h, emb, topo_in)
        for l in self.ups: h = torch.cat([h, sk.pop()], 1); h = l(h, emb, topo_in)
        return self.out(h)


# ================================================================
# Flow Matching  (replaces DDPM noise schedule)
# ================================================================

class FlowMatching:
    """
    Rectified Flow for precipitation downscaling.
    Forward : x_t = (1-t)*x0_noise + t*x1_data
    Target  : v*  = x1_data - x0_noise
    Loss    : MSE(model(x_t, t, cond), v*)
    Sample  : Euler ODE, default 6 steps
    """
    def __init__(self, n_steps=6, cfg_scale=2.5):
        self.n_steps   = n_steps
        self.cfg_scale = cfg_scale

    def get_train_sample(self, x1):
        """Returns x_t, t [B], v_target for training."""
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        # Beta(1.5,1.5) biases toward mid-trajectory — harder to learn
        alpha = torch.distributions.Beta(1.5, 1.5).sample((B,)).to(x1.device)
        t = alpha.view(B, 1, 1, 1)
        x_t = (1 - t) * x0 + t * x1
        v_target = x1 - x0
        return x_t, alpha, v_target

    @torch.no_grad()
    def sample(self, model, x_cond, topo, global_features=None, d2m=None,
               cfg_scale=None, T=1):
        """Euler ODE from noise (t=0) to data (t=1)."""
        cfg = cfg_scale or self.cfg_scale
        B, _, H, W = x_cond.shape
        x  = torch.randn(B, 1, H, W, device=x_cond.device)
        dt = 1.0 / self.n_steps

        for i in range(self.n_steps):
            t_vec = torch.full((B,), i * dt, device=x.device)
            x_in  = torch.cat([x, x_cond], dim=1)
            v_c = model(x_in, t_vec, topo=topo,
                        global_features=global_features, d2m=d2m, T=T)
            if cfg > 1.0:
                x_unc = torch.cat([x, torch.zeros_like(x_cond)], dim=1)
                mask  = torch.ones(B, dtype=torch.bool, device=x.device)
                v_u = model(x_unc, t_vec, topo=topo,
                            global_features=global_features, d2m=d2m,
                            cfg_drop=mask, T=T)
                v = v_u + cfg * (v_c - v_u)
            else:
                v = v_c
            x = x + dt * v

        return x

    @staticmethod
    def loss(v_pred, v_target):
        return F.mse_loss(v_pred, v_target)


# ================================================================
# Physics Guidance
# ================================================================

class PhysicsGuide:
    """
    Hard physical constraints on precipitation (log1p space).
    1. Non-negativity
    2. Dry-cell masking  (India monsoon: dry coarse -> zero fine)
    3. Mass conservation (spatial mean match coarse input)

    Pipeline: pred -> PhysicsGuide.apply -> QDM.apply
    """
    DRY_THRESH_LOG = 0.01

    @staticmethod
    def apply(pred_log: torch.Tensor,
              coarse_log: torch.Tensor,
              enforce_mass: bool = True,
              enforce_dry:  bool = True) -> torch.Tensor:
        pred = pred_log.clamp(min=0.)

        if enforce_dry:
            coarse_mean = coarse_log.mean(dim=[-2,-1], keepdim=True)
            dry_mask    = coarse_mean < PhysicsGuide.DRY_THRESH_LOG
            pred = pred * (~dry_mask).float()

        if enforce_mass:
            pred_phys   = torch.expm1(pred.clamp(0))
            coarse_phys = torch.expm1(coarse_log.clamp(0))
            coarse_up   = F.interpolate(coarse_phys, size=pred.shape[-2:],
                                        mode='bilinear', align_corners=False)
            target_mean = coarse_up.mean(dim=[-2,-1], keepdim=True).clamp(1e-6)
            pred_mean   = pred_phys.mean(dim=[-2,-1], keepdim=True).clamp(1e-6)
            scale       = (target_mean / pred_mean).clamp(0.1, 10.0)
            pred        = torch.log1p((pred_phys * scale).clamp(0))

        return pred


# ================================================================
# QDM
# ================================================================

class QDM:
    def __init__(self, n_quantiles=1000, clip_min=0.):
        self.n = n_quantiles; self.clip = clip_min
        self.q = torch.linspace(0., 1., n_quantiles)
        self.cm = self.co = None; self._fitted = False

    def fit(self, mp, op):
        mp = mp.flatten().float().cpu(); op = op.flatten().float().cpu()
        self.cm = torch.quantile(mp, self.q); self.co = torch.quantile(op, self.q)
        self._fitted = True
        i95 = int(.95*self.n); i99 = int(.99*self.n)
        print(f"[QDM] P95 model={self.cm[i95]:.2f} obs={self.co[i95]:.2f} | "
              f"P99 model={self.cm[i99]:.2f} obs={self.co[i99]:.2f}")

    @torch.no_grad()
    def apply(self, pred):
        assert self._fitted
        dev = pred.device; cm = self.cm.to(dev); co = self.co.to(dev)
        x = pred.flatten().float()
        idx = torch.searchsorted(cm.contiguous(), x.contiguous()).clamp(0, self.n-1)
        return (co[idx]*(x/cm[idx].clamp(1e-6)).clamp(0, 10)).clamp(self.clip).reshape(pred.shape).to(pred.dtype)

    def save(self, p): torch.save({"cm":self.cm,"co":self.co,"n":self.n,"clip":self.clip}, p)

    @classmethod
    def load(cls, p):
        d = torch.load(p, map_location="cpu"); q = cls(d["n"], d["clip"])
        q.cm = d["cm"]; q.co = d["co"]; q._fitted = True; return q
