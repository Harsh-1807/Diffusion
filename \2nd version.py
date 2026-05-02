# -*- coding: utf-8 -*-
"""
Stage 2: Train CorrDiff UNet  —  Joint End-to-End Tuning
=========================================================
DUAL-MODE TRAINER  ──  plug-and-play, 4 × H100

CHANGES vs v2 (the previous version)
──────────────────────────────────────
  * JOINT TUNING  : Regressor unfrozen, trained at LR * REG_LR_SCALE (0.05×)
    - reg added to optimizer as a second param group with lower LR
    - reg wrapped in DDP alongside UNet
    - reg.train() / reg.eval() toggled correctly in train / val phases
    - reg state saved & reloaded in every checkpoint → safe resume
    - grad clip applied across BOTH model and reg parameters
    - mu computation moved INSIDE autocast (no more torch.no_grad())
      so gradients flow back through reg during training
  * All v2 improvements retained:
    T_COND=5, LAMBDA_PCC=0.5, ACCUM_STEPS=4, LR=2e-4,
    UNET_D2M_CH=2 (d2m+var_map_up), cudnn.benchmark, TF32,
    non_blocking, persistent_workers, prefetch_factor=2

Joint-tuning design notes
──────────────────────────
  The regressor (Stage-1) produces mu.  Stage-2 UNet predicts residuals
  around mu.  If mu is spatially wrong, the UNet wastes capacity on
  location error instead of high-frequency texture.  Allowing mu to drift
  slowly (5x LR scale) lets the whole two-stage system align internally.

  EMA covers only the UNet — the regressor is already a slow-moving
  component.  Validation always uses live reg weights + EMA-UNet weights.

  Checkpoint saves: model, reg, optimizer, ema_shadow → fully resumable.
"""

import os
import math
import time
import argparse
import traceback
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

try:
    import albumentations as A
    HAS_ALB = True
except ImportError:
    HAS_ALB = False
    warnings.warn("albumentations not found – pip install albumentations")

from Dataset import UpscaleDataset
from HybridNetwork import CorrDiffRegressor, UNet, FlowMatching, PhysicsGuide, QDM


# ══════════════════════════════════════════════════════════════════════════════
# 0.  MODE TOGGLE
# ══════════════════════════════════════════════════════════════════════════════
TRAIN_MODE = os.environ.get("TRAIN_MODE", "corrdiff_residual")
assert TRAIN_MODE in ("flow_matching", "corrdiff_residual"), \
    f"Bad TRAIN_MODE={TRAIN_MODE!r}"

SIGMA_DATA = 0.1925  # Stage-1 residual std — do NOT change


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PATHS
# ══════════════════════════════════════════════════════════════════════════════
RF_PATH  = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
ORO_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"
D2M_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/era5_aligned_to_rf.nc"
REG_CKPT = "checkpoints/regressor/regressor_best.pth"
CKPT_DIR = "checkpoints/Hybrid/"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
BATCH        = 16
ACCUM_STEPS  = 4              # effective batch = 64/GPU
LR           = 2e-4           # UNet LR
REG_LR_SCALE = 0.05           # reg LR = LR * 0.05  (20x smaller)
MIN_LR       = LR * 0.01
EPOCHS       = 1500
PATIENCE     = 200
T_COND       = 5
PRECIP_CH    = 0
BASE_CH      = 256
CHANNEL_MULT = (1, 2, 2, 4)
NRB          = 2
DROPOUT      = 0.15
FM_STEPS     = 6
CFG_SCALE    = 1.5
P_CFG_DROP   = 0.15
LAMBDA_SPEC  = 0.2
LAMBDA_PCC   = 0.5
WEIGHT_DECAY = 1e-3
GRAD_CLIP    = 1.0
EMA_DECAY    = 0.9995
N_ENS        = 2
REG_IN_CH    = 2              # regressor coarse input channels
REG_D2M_CH   = 1              # regressor d2m channels (raw d2m only)
D2M_CH       = 1              # raw d2m channels in dataset
UNET_D2M_CH  = 2              # UNet d2m channels = d2m + var_map_up
TOPO_CH      = 3
GLOBAL_DIM   = 2
UNET_IN_CH   = 1 + 1 + T_COND    # noisy_res + mu + temporal_frames
DS_FACTOR    = 4


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EDM NOISE SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
def build_edm_schedule(n_steps, sigma_min=0.002, sigma_data=SIGMA_DATA, rho=7.0):
    sigma_max = 2.0 * sigma_data
    steps = torch.arange(n_steps, dtype=torch.float32) / max(n_steps - 1, 1)
    return (sigma_max**(1/rho) + steps*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho

_EDM_CPU = build_edm_schedule(FM_STEPS)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EMA  (UNet only — reg is its own slow-moving component)
# ══════════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay  = decay
        self.shadow = {k: v.clone().detach().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1-self.decay)

    def apply_to(self, model):
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in self.shadow.items()})

    def restore(self, model, state):
        model.load_state_dict(state)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def _make_aug_pipeline():
    if not HAS_ALB:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                           rotate_limit=10, border_mode=0, p=0.4),
        A.ElasticTransform(alpha=1.2, sigma=50.0, p=0.25),
        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.1,
                                   contrast_limit=0.1, p=0.4),
        A.GaussNoise(var_limit=(1e-5, 5e-4), p=0.25),
    ], additional_targets={"topo": "image", "d2m": "image"})

_AUG_PIPELINE = _make_aug_pipeline()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TOPO / COARSE / D2M HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def compute_slope_aspect(elev):
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32,
                       device=elev.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32,
                       device=elev.device).view(1,1,3,3)
    e  = elev.float()
    dx = F.conv2d(e, kx, padding=1); dy = F.conv2d(e, ky, padding=1)
    slope  = torch.sqrt(dx**2 + dy**2 + 1e-8)
    aspect = torch.atan2(dy, dx)
    def norm(t):
        mn=t.amin(); mx=t.amax(); return 2*(t-mn)/(mx-mn+1e-8)-1
    return torch.cat([norm(e), norm(slope), norm(aspect)], dim=1)

def expand_topo(topo_1ch):
    return torch.cat([compute_slope_aspect(topo_1ch[i:i+1])
                      for i in range(topo_1ch.shape[0])], dim=0)

def build_coarse_input(coarse, var_map):
    Hc, Wc = coarse.shape[-2], coarse.shape[-1]
    return torch.cat([coarse, F.adaptive_avg_pool2d(var_map, (Hc, Wc))], dim=1)

def build_unet_d2m(d2m, var_map, fine_size):
    """Concatenate d2m [B,1] + var_map_up [B,1] → [B,2,Hf,Wf] for the UNet."""
    var_up = F.interpolate(var_map, size=fine_size, mode='bilinear', align_corners=False)
    return torch.cat([d2m, var_up], dim=1) if d2m is not None else var_up


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TEMPORAL CONDITIONING
# ══════════════════════════════════════════════════════════════════════════════
def build_temporal_cond(batch, dev, n_frames=T_COND):
    if "tc_frames" in batch:
        tc = batch["tc_frames"].to(dev, non_blocking=True)
        if tc.shape[1] >= n_frames:
            return tc[:, :n_frames]
        pad = torch.zeros(tc.shape[0], n_frames-tc.shape[1], *tc.shape[2:], device=dev)
        return torch.cat([tc, pad], dim=1)
    coarse    = batch["coarse"].to(dev, non_blocking=True)
    coarse_up = F.interpolate(coarse, scale_factor=4, mode='bilinear', align_corners=False)
    return coarse_up.expand(-1, n_frames, -1, -1)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  AUXILIARY LOSSES
# ══════════════════════════════════════════════════════════════════════════════
def spectral_loss(pred, target):
    P=torch.fft.rfft2(pred.float()); T=torch.fft.rfft2(target.float()); eps=1e-8
    return ((torch.log(P.abs()+eps)-torch.log(T.abs()+eps))**2).mean()

def hybrid_sigma_loss(pred, target, sigma, alpha=0.8):
    eps=1e-6; p=pred.float(); t=target.float()
    mae=torch.abs(p-t)
    scaled_mae=(mae/(sigma+eps)).mean()
    w=(torch.sigmoid((t-0.015)*150)*3.5 + torch.sigmoid((t-0.08)*50)*5.0 +
       torch.sigmoid((t-0.25)*20)*6.0   + torch.sigmoid((t-0.5)*10)*7.0 +
       (1.-torch.sigmoid((t-0.015)*150))*0.6)
    return alpha*scaled_mae + (1-alpha)*(w*mae).mean()

def pcc_loss(pred, target):
    B=pred.shape[0]
    p=pred.view(B,-1).float(); t=target.view(B,-1).float()
    pm=p-p.mean(dim=1,keepdim=True); tm=t-t.mean(dim=1,keepdim=True)
    return (1-(pm*tm).sum(dim=1)/(
        torch.sqrt((pm**2).sum(dim=1))*torch.sqrt((tm**2).sum(dim=1))+1e-8)).mean()

@torch.no_grad()
def weighted_pcc(pred, target, lat_w=None):
    B=pred.shape[0]
    p=pred.float().view(B,-1); t=target.float().view(B,-1)
    pm=p.mean(dim=1,keepdim=True); tm=t.mean(dim=1,keepdim=True)
    pm_c=p-pm; tm_c=t-tm
    r=(pm_c*tm_c).sum(dim=1)/(
        torch.sqrt((pm_c**2).sum(dim=1))*torch.sqrt((tm_c**2).sum(dim=1))+1e-8)
    return r.mean().item()

@torch.no_grad()
def crps_ensemble(samples, target):
    N=samples.shape[0]
    mae=(samples-target.unsqueeze(0)).abs().mean(0)
    pair=(samples.unsqueeze(0)-samples.unsqueeze(1)).abs()
    return (mae-0.5/N/(N-1)*pair.sum([0,1])).mean().item()

@torch.no_grad()
def psd_tail_ratio(pred, target, hff=0.3):
    P=torch.fft.rfft2(pred.float()).abs(); T=torch.fft.rfft2(target.float()).abs()
    c=int((1-hff)*P.shape[-1])
    return (P[...,c:].mean()/(T[...,c:].mean()+1e-8)).item()

@torch.no_grad()
def fractions_skill_score(pred, target, threshold=0.5, window=5):
    p_bin=(pred>threshold).float(); t_bin=(target>threshold).float()
    p_frac=F.avg_pool2d(p_bin,kernel_size=window,stride=1,padding=window//2)
    t_frac=F.avg_pool2d(t_bin,kernel_size=window,stride=1,padding=window//2)
    mse=((p_frac-t_frac)**2).mean(dim=[-1,-2])
    ref=(p_frac**2).mean(dim=[-1,-2])+(t_frac**2).mean(dim=[-1,-2])
    return (1.-mse/(ref+1e-8)).mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DDP SETUP
# ══════════════════════════════════════════════════════════════════════════════
def setup():
    rank=int(os.environ.get("RANK",0)); ws=int(os.environ.get("WORLD_SIZE",1))
    lr_=int(os.environ.get("LOCAL_RANK",0))
    if ws>1: dist.init_process_group("nccl"); torch.cuda.set_device(lr_)
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    return rank,ws,lr_,torch.device(f"cuda:{lr_}" if torch.cuda.is_available() else "cpu")

def ar(t, ws):
    if ws>1 and dist.is_initialized(): dist.all_reduce(t,op=dist.ReduceOp.SUM); t/=ws
    return t


# ══════════════════════════════════════════════════════════════════════════════
# 10.  REGRESSOR LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_regressor(ckpt_path, dev, freeze=False):
    """
    freeze=False (default) → joint tuning: gradients flow through reg.
    freeze=True            → frozen inference / ablation baseline.
    """
    ck  = torch.load(ckpt_path, map_location=dev)
    reg = CorrDiffRegressor(
        in_channels=ck.get("reg_in_channels", REG_IN_CH), out_channels=1,
        base_channels=64, channel_mult=(1,2,4), num_blocks=2,
        global_dim=GLOBAL_DIM, topo_channels=TOPO_CH,
        d2m_channels=ck.get("d2m_channels", REG_D2M_CH),
        use_d2m=ck.get("use_d2m", True),
    ).to(dev)
    state={k.replace("module.",""): v for k,v in ck["model_state_dict"].items()}
    reg.load_state_dict(state)
    for p in reg.parameters():
        p.requires_grad_(not freeze)
    return reg


# ══════════════════════════════════════════════════════════════════════════════
# 11.  DDIM SAMPLER  (V2)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def ddim_sample(raw_model, mu, tc, tp, gf, d2m_unet, edm_schedule, dev):
    B=mu.shape[0]; x_t=torch.randn_like(mu)*SIGMA_DATA
    sigmas=edm_schedule.to(dev)
    for i, sigma_cur in enumerate(sigmas):
        s_cur =sigma_cur.view(1,1,1,1)
        c_in  =1./torch.sqrt(s_cur**2+SIGMA_DATA**2)
        c_out =s_cur*SIGMA_DATA/torch.sqrt(s_cur**2+SIGMA_DATA**2)
        c_skip=SIGMA_DATA**2/(s_cur**2+SIGMA_DATA**2)
        c_n   =(sigma_cur.log()/4).expand(B)
        x_in  =torch.cat([x_t,mu,tc],dim=1)
        D_pred=raw_model(c_in*x_in,c_n,topo=tp,global_features=gf,
                         d2m=d2m_unet,T=T_COND)
        x0_hat=c_skip*x_t[:,:1]+c_out*D_pred
        if i<len(sigmas)-1:
            sigma_next=sigmas[i+1].view(1,1,1,1)
            x_t=x0_hat+sigma_next*(x_t-x0_hat)/s_cur.clamp(min=1e-8)
        else:
            x_t=x0_hat
    return x_t


# ══════════════════════════════════════════════════════════════════════════════
# 12.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train():
    rank,ws,lr_,dev=setup()
    os.makedirs(CKPT_DIR, exist_ok=True)

    SAVE    =os.path.join(CKPT_DIR,f"unet_{TRAIN_MODE}_sigma{SIGMA_DATA:.3f}_best.pth")
    LATEST  =os.path.join(CKPT_DIR,f"unet_{TRAIN_MODE}_sigma{SIGMA_DATA:.3f}_latest.pth")
    QDM_PATH=os.path.join(CKPT_DIR,f"qdm_{TRAIN_MODE}.pth")

    edm_schedule=_EDM_CPU.clone()

    # ── dataset ───────────────────────────────────────────────────────────────
    ds=UpscaleDataset(RF_PATH,ORO_PATH,d2m_file=D2M_PATH,
                      split="train",normalize=True,device="cpu")
    n=len(ds); trn=int(0.70*n); van=int(0.10*n)
    _lkw=dict(num_workers=8, pin_memory=True,
              persistent_workers=True, prefetch_factor=2, drop_last=True)
    trl=DataLoader(Subset(ds,range(0,trn)),       BATCH, shuffle=True,  **_lkw)
    val=DataLoader(Subset(ds,range(trn,trn+van)), BATCH, shuffle=False,
                   num_workers=8, pin_memory=True,
                   persistent_workers=True, prefetch_factor=2)

    # ── models ────────────────────────────────────────────────────────────────
    # Regressor: unfrozen for joint tuning
    reg=load_regressor(REG_CKPT, dev, freeze=False)

    model=UNet(
        in_channels=UNET_IN_CH, out_channels=1, base_channels=BASE_CH,
        channel_mult=CHANNEL_MULT, num_res_blocks=NRB, dropout=DROPOUT,
        global_dim=GLOBAL_DIM, use_bottleneck_attention=True,
        topo_channels=TOPO_CH, use_d2m=True,
        d2m_channels=UNET_D2M_CH,   # 2: d2m + var_map_up
        temporal_frames=T_COND,
    ).to(dev)

    # Wrap BOTH in DDP — critical: reg must also be DDP for NCCL sync
    if ws>1:
        reg  =nn.parallel.DistributedDataParallel(
                  reg,   device_ids=[lr_], find_unused_parameters=False)
        model=nn.parallel.DistributedDataParallel(
                  model, device_ids=[lr_], find_unused_parameters=False)

    raw     = model.module if ws>1 else model   # bare UNet for EMA / sampling
    raw_reg = reg.module   if ws>1 else reg     # bare reg for checkpointing

    ema=EMA(raw, decay=EMA_DECAY)

    # ── optimizer: two param groups, differential LR ──────────────────────────
    # UNet  : LR = 2e-4
    # Reg   : LR = 2e-4 * 0.05 = 1e-5  (20x smaller → slow drift)
    opt=AdamW([
        {"params": model.parameters(), "lr": LR},
        {"params": reg.parameters(),   "lr": LR * REG_LR_SCALE},
    ], weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))

    scaler=GradScaler(device=dev.type)
    # CosineAnnealingWarmRestarts respects per-group base_lrs
    sched=CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=1, eta_min=MIN_LR)

    if TRAIN_MODE=="flow_matching":
        fm=FlowMatching(n_steps=FM_STEPS,cfg_scale=CFG_SCALE); loss_label="L_flow  "
    else:
        fm=None; loss_label="L_denoise"

    start=0; best_wpcc=0.0; no_improve=0

    # ── resume ────────────────────────────────────────────────────────────────
    if os.path.exists(LATEST):
        ck=torch.load(LATEST, map_location=dev)
        # UNet weights
        try:
            raw.load_state_dict({k.replace("module.",""): v
                                 for k,v in ck["model_state_dict"].items()})
        except RuntimeError as e:
            if rank==0: print(f"[RESUME] UNet partial load: {e}")
        # Regressor weights (absent in older frozen-reg checkpoints → silent skip)
        if "reg_state_dict" in ck:
            try:
                raw_reg.load_state_dict({k.replace("module.",""): v
                                         for k,v in ck["reg_state_dict"].items()})
                if rank==0: print("[RESUME] Reg weights restored from joint checkpoint")
            except RuntimeError as e:
                if rank==0: print(f"[RESUME] Reg partial load: {e}")
        else:
            if rank==0: print("[RESUME] No reg_state_dict found — using REG_CKPT weights")
        # Optimizer (may mismatch if old run had 1 param group)
        try:
            opt.load_state_dict(ck["optimizer_state_dict"])
        except (ValueError, KeyError) as e:
            if rank==0: print(f"[RESUME] Optimizer reset (param groups changed): {e}")
        start=ck.get("epoch",0)+1; best_wpcc=ck.get("wpcc",0.0)
        no_improve=ck.get("no_improve",0)
        if "ema_shadow" in ck:
            ema.shadow={k:v.to(dev) for k,v in ck["ema_shadow"].items()}
        if rank==0:
            print(f"[RESUME] ep={start}  best_wPCC={best_wpcc:.4f}")

    if rank==0:
        np_u=sum(p.numel() for p in model.parameters() if p.requires_grad)
        np_r=sum(p.numel() for p in reg.parameters()   if p.requires_grad)
        print(f"[MODEL]  UNet {np_u/1e6:.2f}M  Reg {np_r/1e6:.2f}M  "
              f"mode={TRAIN_MODE}  sigma_data={SIGMA_DATA}")
        print(f"[OPTIM]  LR_unet={LR}  LR_reg={LR*REG_LR_SCALE:.2e}  "
              f"ACCUM={ACCUM_STEPS}  eff_batch={BATCH*ACCUM_STEPS}  LAMBDA_PCC={LAMBDA_PCC}")
        print(f"[COND]   UNET_D2M_CH={UNET_D2M_CH} (d2m+var_map_up)  T_COND={T_COND}")
        hdr=(f"{'Ep':>5}|{loss_label:>10}|{'L_spec':>8}|{'L_pcc':>8}"
             f"|{'Val':>8}|{'wPCC':>7}|{'CRPS':>8}|{'PSD_r':>7}|{'FSS':>6}"
             f"|{'LR_u':>9}|{'LR_r':>9}")
        print(hdr); print("-"*len(hdr))

    lat_w=None

    for ep in range(start, EPOCHS):
        model.train(); reg.train()        # ← BOTH in train mode
        t0=time.time(); sum_ml=sum_sp=sum_pc=nb=0.
        opt.zero_grad(set_to_none=True)

        for step, b in enumerate(trl, 1):
            try:
                # ── non-blocking GPU transfers ────────────────────────────
                fp      =b["fine"].to(dev,non_blocking=True)[:,PRECIP_CH:PRECIP_CH+1]
                topo_1ch=b["topo"].to(dev,non_blocking=True)
                xi_raw  =b["coarse"].to(dev,non_blocking=True)
                d2m     =b["d2m"].to(dev,non_blocking=True) if "d2m" in b else None
                var_map =b["var_map"].to(dev,non_blocking=True)
                gf      =torch.stack([b["doy"],b["hour"]],1).float().to(dev,non_blocking=True)
                tc      =build_temporal_cond(b, dev)

                # ── GPU-native augmentation ───────────────────────────────
                if torch.rand(1).item()<0.5:
                    fp=fp.flip(-1); topo_1ch=topo_1ch.flip(-1)
                    var_map=var_map.flip(-1)
                    if d2m is not None: d2m=d2m.flip(-1)
                if torch.rand(1).item()<0.5:
                    fp=fp.flip(-2); topo_1ch=topo_1ch.flip(-2)
                    var_map=var_map.flip(-2)
                    if d2m is not None: d2m=d2m.flip(-2)

                xi_raw  =F.avg_pool2d(fp, kernel_size=DS_FACTOR, stride=DS_FACTOR)
                tp      =expand_topo(topo_1ch)
                xi      =build_coarse_input(xi_raw, var_map)
                d2m_unet=build_unet_d2m(d2m, var_map, fine_size=fp.shape[-2:])
                cfg_drop=(torch.rand(fp.shape[0],device=dev)<P_CFG_DROP)

                # ── full forward — reg inside autocast, gradients enabled ─
                # IMPORTANT: no torch.no_grad() around reg — gradients must
                # flow back so AdamW can update reg at LR * REG_LR_SCALE
                with autocast(device_type=dev.type):
                    mu      =reg(xi, topo=tp, global_features=gf, d2m=d2m)
                    residual=fp-mu

                    # ── V1: FLOW MATCHING ─────────────────────────────────
                    if TRAIN_MODE=="flow_matching":
                        x_t,t_vec,v_star=fm.get_train_sample(residual)
                        t_bcast =t_vec.view(-1,1,1,1)
                        x_in    =torch.cat([x_t,mu,tc],dim=1)
                        v_pred  =model(x_in,t_vec,topo=tp,global_features=gf,
                                       cfg_drop=cfg_drop,d2m=d2m_unet,T=T_COND)
                        l_main  =fm.loss(v_pred,v_star)
                        fine_hat=x_t+(1-t_bcast)*v_pred+mu

                    # ── V2: CORRDIFF RESIDUAL ─────────────────────────────
                    else:
                        idx    =torch.randint(0,len(edm_schedule),(fp.shape[0],))
                        sigma_t=edm_schedule[idx].to(dev).view(-1,1,1,1)
                        eps    =torch.randn_like(residual)
                        x_t    =residual+sigma_t*eps
                        c_in   =1./torch.sqrt(sigma_t**2+SIGMA_DATA**2)
                        c_out  =sigma_t*SIGMA_DATA/torch.sqrt(sigma_t**2+SIGMA_DATA**2)
                        c_skip =SIGMA_DATA**2/(sigma_t**2+SIGMA_DATA**2)
                        c_n    =(sigma_t.log()/4).view(fp.shape[0])
                        x_in   =torch.cat([x_t,mu,tc],dim=1)
                        D_pred =model(c_in*x_in,c_n,topo=tp,global_features=gf,
                                      cfg_drop=cfg_drop,d2m=d2m_unet,T=T_COND)
                        x0_pred=c_skip*x_t[:,:1]+c_out*D_pred
                        l_main =hybrid_sigma_loss(x0_pred,residual,sigma_t,alpha=0.8)
                        fine_hat=x0_pred+mu

                    l_spec=spectral_loss(fine_hat,fp)
                    l_pcc =pcc_loss(fine_hat,fp)
                    loss  =(l_main+LAMBDA_SPEC*l_spec+LAMBDA_PCC*l_pcc)/ACCUM_STEPS

                scaler.scale(loss).backward()

                if step%ACCUM_STEPS==0:
                    scaler.unscale_(opt)
                    # ← clip across BOTH UNet and reg to prevent reg blowup
                    all_params=list(model.parameters())+list(reg.parameters())
                    nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                    ema.update(raw)    # EMA on UNet only — reg excluded

                with torch.no_grad():
                    sum_ml+=l_main.item(); sum_sp+=l_spec.item()
                    sum_pc+=l_pcc.item();  nb+=1

            except Exception:
                if rank==0: traceback.print_exc()
                raise

        sched.step()
        te_ml=ar(torch.tensor(sum_ml/max(nb,1),device=dev),ws).item()
        te_sp=ar(torch.tensor(sum_sp/max(nb,1),device=dev),ws).item()
        te_pc=ar(torch.tensor(sum_pc/max(nb,1),device=dev),ws).item()
        lr_u=opt.param_groups[0]["lr"]
        lr_r=opt.param_groups[1]["lr"]

        # ── VALIDATION  (EMA-UNet + live reg weights) ─────────────────────
        live_state=deepcopy(raw.state_dict())
        ema.apply_to(raw)
        model.eval(); reg.eval()          # ← BOTH in eval mode

        v_loss=torch.tensor(0.,device=dev); vn=torch.tensor(0,device=dev,dtype=torch.long)
        pcc_sum=crps_sum=psd_sum=fss_sum=0.; vb=0

        with torch.no_grad():
            for b in val:
                fp     =b["fine"].to(dev,non_blocking=True)[:,PRECIP_CH:PRECIP_CH+1]
                tp     =expand_topo(b["topo"].to(dev,non_blocking=True))
                gf     =torch.stack([b["doy"],b["hour"]],1).float().to(dev,non_blocking=True)
                xi_raw =b["coarse"].to(dev,non_blocking=True)
                var_map=b["var_map"].to(dev,non_blocking=True)
                d2m    =b["d2m"].to(dev,non_blocking=True) if "d2m" in b else None
                tc     =build_temporal_cond(b,dev)
                xi     =build_coarse_input(xi_raw,var_map)
                d2m_unet=build_unet_d2m(d2m,var_map,fine_size=fp.shape[-2:])

                with autocast(device_type=dev.type):
                    mu=reg(xi,topo=tp,global_features=gf,d2m=d2m)
                residual=fp-mu

                if TRAIN_MODE=="flow_matching":
                    x_t,t_vec,v_star=fm.get_train_sample(residual)
                    x_in=torch.cat([x_t,mu,tc],dim=1)
                    with autocast(device_type=dev.type):
                        v_pred=model(x_in,t_vec,topo=tp,global_features=gf,
                                     d2m=d2m_unet,T=T_COND)
                        l_val =fm.loss(v_pred,v_star)
                else:
                    idx    =torch.randint(0,len(edm_schedule),(fp.shape[0],))
                    sigma_t=edm_schedule[idx].to(dev).view(-1,1,1,1)
                    eps    =torch.randn_like(residual); x_t=residual+sigma_t*eps
                    c_in   =1./torch.sqrt(sigma_t**2+SIGMA_DATA**2)
                    c_out  =sigma_t*SIGMA_DATA/torch.sqrt(sigma_t**2+SIGMA_DATA**2)
                    c_skip =SIGMA_DATA**2/(sigma_t**2+SIGMA_DATA**2)
                    c_n    =(sigma_t.log()/4).view(fp.shape[0])
                    x_in   =torch.cat([x_t,mu,tc],dim=1)
                    with autocast(device_type=dev.type):
                        D_pred =model(c_in*x_in,c_n,topo=tp,global_features=gf,
                                      d2m=d2m_unet,T=T_COND)
                        x0_pred=c_skip*x_t[:,:1]+c_out*D_pred
                        l_val  =hybrid_sigma_loss(x0_pred,residual,sigma_t,alpha=0.8)

                v_loss+=l_val*fp.shape[0]; vn+=fp.shape[0]

                samples=[]
                for _ in range(N_ENS):
                    if TRAIN_MODE=="flow_matching":
                        x_cond=torch.cat([mu,tc],dim=1)
                        s=fm.sample(raw,x_cond,topo=tp,global_features=gf,
                                    d2m=d2m_unet,cfg_scale=CFG_SCALE,T=T_COND)+mu
                    else:
                        s=ddim_sample(raw,mu,tc,tp,gf,d2m_unet,edm_schedule,dev)+mu
                    s=PhysicsGuide.apply(s,xi_raw,enforce_mass=True,enforce_dry=True)
                    samples.append(s)

                samples_t=torch.stack(samples); best_s=samples_t.mean(0)
                pcc_sum +=weighted_pcc(best_s,fp,lat_w)
                crps_sum+=crps_ensemble(samples_t,fp)
                psd_sum +=psd_tail_ratio(best_s,fp)
                fss_sum +=fractions_skill_score(best_s,fp,threshold=0.5,window=5)
                vb+=1

        ema.restore(raw,live_state)
        model.train(); reg.train()        # ← restore BOTH to train mode

        if ws>1 and dist.is_initialized():
            dist.barrier()
            for t__ in [v_loss,vn]: dist.all_reduce(t__,op=dist.ReduceOp.SUM)
            dist.barrier()

        vw   =(v_loss/vn.clamp(1)).item()
        wpcc =pcc_sum/max(vb,1); crps_v=crps_sum/max(vb,1)
        psd_r=psd_sum/max(vb,1); fss_v =fss_sum/max(vb,1)
        el   =time.time()-t0

        if rank==0:
            star=" ★" if wpcc>best_wpcc else ""
            print(f"{ep:>5}|{te_ml:>10.5f}|{te_sp:>8.5f}|{te_pc:>8.5f}"
                  f"|{vw:>8.4f}|{wpcc:>7.4f}|{crps_v:>8.4f}|{psd_r:>7.3f}|{fss_v:>6.3f}"
                  f"|{lr_u:>9.2e}|{lr_r:>9.2e}  [{el:.0f}s]{star}")

            # ── checkpoint: save UNet + Reg + optimizer + EMA ─────────────
            ck_base={
                # UNet
                "epoch": ep,
                "model_state_dict": raw.state_dict(),
                "ema_shadow": ema.shadow,
                # Regressor  ← NEW: saved alongside UNet for consistent reload
                "reg_state_dict": raw_reg.state_dict(),
                # Optimizer (two param groups)
                "optimizer_state_dict": opt.state_dict(),
                # Metrics
                "val_loss": vw, "wpcc": wpcc, "crps": crps_v,
                "psd_tail_ratio": psd_r, "no_improve": no_improve,
                # Metadata for inference reload
                "train_mode": TRAIN_MODE, "sigma_data": SIGMA_DATA,
                "unet_in_channels": UNET_IN_CH, "t_cond": T_COND,
                "d2m_channels": UNET_D2M_CH,
                "reg_d2m_channels": REG_D2M_CH,
            }
            torch.save(ck_base, LATEST)

            if wpcc>best_wpcc:
                best_wpcc=wpcc; no_improve=0
                torch.save(ck_base, SAVE)
                print(f"  ★ BEST  val_loss={vw:.4f}  wPCC={wpcc:.4f}  "
                      f"CRPS={crps_v:.4f}  FSS={fss_v:.3f}")
                if (ep+1)%50==0:
                    torch.save(ck_base, os.path.join(
                        CKPT_DIR,
                        f"unet_{TRAIN_MODE}_ep{ep+1:04d}_wpcc{wpcc:.4f}.pth"))
            else:
                no_improve+=1

        if no_improve>=PATIENCE:
            if rank==0: print(f"\n⚠  Early stop ep={ep+1}  best_wPCC={best_wpcc:.4f}")
            break

    # ── QDM calibration ───────────────────────────────────────────────────────
    if rank==0:
        print("\n[QDM] fitting on val set …")
        qdm=QDM(n_quantiles=1000)
        ck_best=torch.load(SAVE, map_location=dev)
        raw.load_state_dict({k.replace("module.",""): v
                             for k,v in ck_best["model_state_dict"].items()})
        if "reg_state_dict" in ck_best:
            raw_reg.load_state_dict({k.replace("module.",""): v
                                     for k,v in ck_best["reg_state_dict"].items()})
        if "ema_shadow" in ck_best:
            ema.shadow={k:v.to(dev) for k,v in ck_best["ema_shadow"].items()}
            ema.apply_to(raw)
        model.eval(); reg.eval()
        all_pred,all_obs=[],[]
        with torch.no_grad():
            for b in val:
                fp     =b["fine"].to(dev,non_blocking=True)[:,PRECIP_CH:PRECIP_CH+1]
                tp     =expand_topo(b["topo"].to(dev,non_blocking=True))
                gf     =torch.stack([b["doy"],b["hour"]],1).float().to(dev,non_blocking=True)
                xi_raw =b["coarse"].to(dev,non_blocking=True)
                var_map=b["var_map"].to(dev,non_blocking=True)
                d2m    =b["d2m"].to(dev,non_blocking=True) if "d2m" in b else None
                tc     =build_temporal_cond(b,dev)
                xi     =build_coarse_input(xi_raw,var_map)
                d2m_unet=build_unet_d2m(d2m,var_map,fine_size=fp.shape[-2:])
                with autocast(device_type=dev.type):
                    mu=reg(xi,topo=tp,global_features=gf,d2m=d2m)
                if TRAIN_MODE=="flow_matching":
                    x_cond=torch.cat([mu,tc],dim=1)
                    s=fm.sample(raw,x_cond,topo=tp,global_features=gf,
                                d2m=d2m_unet,cfg_scale=CFG_SCALE,T=T_COND)+mu
                else:
                    s=ddim_sample(raw,mu,tc,tp,gf,d2m_unet,edm_schedule,dev)+mu
                pred=PhysicsGuide.apply(s,xi_raw,enforce_mass=True,enforce_dry=True)
                all_pred.append(pred.cpu()); all_obs.append(fp.cpu())
        qdm.fit(torch.cat(all_pred),torch.cat(all_obs))
        qdm.save(QDM_PATH); print(f"[QDM] saved → {QDM_PATH}")

    if ws>1 and dist.is_initialized(): dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════════
# 13.  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
class CorrDiffInference:
    def __init__(self, reg_ckpt, unet_ckpt, qdm_ckpt, device,
                 cfg_scale=CFG_SCALE, n_ens=8):
        self.dev=device; self.cfg_scale=cfg_scale; self.n_ens=n_ens

        ck=torch.load(unet_ckpt, map_location=device)

        # Load regressor — prefer joint checkpoint's reg_state_dict
        self.reg=load_regressor(reg_ckpt, device, freeze=True)
        if "reg_state_dict" in ck:
            self.reg.load_state_dict(
                {k.replace("module.",""): v for k,v in ck["reg_state_dict"].items()})
        self.reg.eval()

        # Load UNet
        tc=ck.get("t_cond",T_COND); self.tc=tc
        self.train_mode=ck.get("train_mode","flow_matching")
        self.fm=FlowMatching(n_steps=FM_STEPS,cfg_scale=cfg_scale)
        self.edm_sched=build_edm_schedule(FM_STEPS,
                                           sigma_data=ck.get("sigma_data",SIGMA_DATA))
        unet_d2m_ch=ck.get("d2m_channels",UNET_D2M_CH)
        self.unet=UNet(
            in_channels=ck.get("unet_in_channels",UNET_IN_CH),out_channels=1,
            base_channels=BASE_CH,channel_mult=CHANNEL_MULT,num_res_blocks=NRB,dropout=0.,
            global_dim=GLOBAL_DIM,topo_channels=TOPO_CH,use_d2m=True,
            d2m_channels=unet_d2m_ch,temporal_frames=tc,
        ).to(device)
        state=(
            {k:v.to(device) for k,v in ck["ema_shadow"].items()} if "ema_shadow" in ck
            else {k.replace("module.",""): v for k,v in ck["model_state_dict"].items()})
        self.unet.load_state_dict(state); self.unet.eval()
        self.qdm=QDM.load(qdm_ckpt) if qdm_ckpt and os.path.exists(qdm_ckpt) else None

    @torch.no_grad()
    def predict(self, coarse, var_map, topo, d2m, doy, hour, tc_frames=None):
        dev=self.dev
        coarse=coarse.to(dev); var_map=var_map.to(dev)
        topo=topo.to(dev);     d2m=d2m.to(dev)
        gf=torch.stack([doy.to(dev),hour.to(dev)],dim=1).float()
        xi=build_coarse_input(coarse,var_map); tp=expand_topo(topo)
        if tc_frames is None:
            coarse_up=F.interpolate(coarse,scale_factor=4,mode='bilinear',align_corners=False)
            tc_frames=coarse_up.expand(-1,self.tc,-1,-1).to(dev)
        else:
            tc_frames=tc_frames.to(dev)
        mu      =self.reg(xi,topo=tp,global_features=gf,d2m=d2m)
        fine_sz =mu.shape[-2:]
        d2m_unet=build_unet_d2m(d2m,var_map,fine_size=fine_sz)
        samples=[]
        for _ in range(self.n_ens):
            if self.train_mode=="flow_matching":
                x_cond=torch.cat([mu,tc_frames],dim=1)
                s=self.fm.sample(self.unet,x_cond,topo=tp,global_features=gf,
                                 d2m=d2m_unet,cfg_scale=self.cfg_scale,T=self.tc)+mu
            else:
                s=ddim_sample(self.unet,mu,tc_frames,tp,gf,d2m_unet,
                              self.edm_sched,dev)+mu
            s=PhysicsGuide.apply(s,coarse,enforce_mass=True,enforce_dry=True)
            if self.qdm is not None: s=self.qdm.apply(s)
            samples.append(s)
        samples=torch.stack(samples)
        return {"mean":samples.mean(0),"std":samples.std(0),"samples":samples,"mu":mu}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",         default=None,
                        choices=["flow_matching","corrdiff_residual"])
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--batch",        type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--reg_lr_scale", type=float, default=None,
                        help="Regressor LR = LR * reg_lr_scale  (default 0.05)")
    parser.add_argument("--dropout",      type=float, default=None)
    parser.add_argument("--patience",     type=int,   default=None)
    args=parser.parse_args()
    if args.mode         is not None and "TRAIN_MODE" not in os.environ:
        TRAIN_MODE=args.mode
    if args.epochs       is not None: EPOCHS=args.epochs
    if args.batch        is not None: BATCH=args.batch
    if args.lr           is not None: LR=args.lr
    if args.reg_lr_scale is not None: REG_LR_SCALE=args.reg_lr_scale
    if args.dropout      is not None: DROPOUT=args.dropout
    if args.patience     is not None: PATIENCE=args.patience
    train()
