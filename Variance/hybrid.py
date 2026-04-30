# -*- coding: utf-8 -*-
"""
Stage 2: Train CorrDiff UNet  (Flow-Matching Corrector  OR  CorrDiff Residual)
===============================================================================
DUAL-MODE TRAINER  ──  plug-and-play, 4 × H100, anti-overfit

TRAIN_MODE = "flow_matching"      V1 – Rectified-flow ODE corrector (original)
TRAIN_MODE = "corrdiff_residual"  V2 – EDM-style σ-weighted denoising (NVIDIA)

Switch via env var or CLI:
  TRAIN_MODE=corrdiff_residual torchrun --nproc_per_node=4 stage2_train.py
  python stage2_train.py --mode corrdiff_residual --epochs 200

Augmentation note (d2m @ 25 km):
  Dataset stores d2m as [1,H,W] (ERA5 25-km upsampled to fine res).
  We apply the SAME geometric transform to fp + topo + d2m simultaneously.
  coarse [1,H/4,W/4] is RE-DERIVED from augmented fp via avg_pool2d so the
  physical downscaling relationship is always preserved.
  var_map and tc_frames are temporal statistics → NOT augmented.
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
    warnings.warn("albumentations not found. pip install albumentations opencv-python-headless")

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
ACCUM_STEPS  = 1
LR           = 1e-4
MIN_LR       = LR * 0.01
EPOCHS       = 1500
PATIENCE     = 200
T_COND       = 3
PRECIP_CH    = 0
BASE_CH      = 256
CHANNEL_MULT = (1, 2, 2, 4)
NRB          = 2
DROPOUT      = 0.15
FM_STEPS     = 6
CFG_SCALE    = 1.5
P_CFG_DROP   = 0.15
LAMBDA_SPEC  = 0.2
LAMBDA_PCC   = 0.1
WEIGHT_DECAY = 1e-3
GRAD_CLIP    = 1.0
EMA_DECAY    = 0.9995
N_ENS        = 2
REG_IN_CH    = 2
D2M_CH       = 1
TOPO_CH      = 3
GLOBAL_DIM   = 2
UNET_IN_CH   = 1 + 1 + T_COND   # noisy_res + mu + temporal_frames
DS_FACTOR    = 4                 # downscale factor (must match Dataset)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EDM NOISE SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
def build_edm_schedule(n_steps, sigma_min=0.002, sigma_data=SIGMA_DATA, rho=7.0):
    sigma_max = 2.0 * sigma_data
    steps = torch.arange(n_steps, dtype=torch.float32) / max(n_steps - 1, 1)
    return (sigma_max**(1/rho) + steps*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho

_EDM_CPU = build_edm_schedule(FM_STEPS)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EMA
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
# 5.  AUGMENTATION  ──  CONSISTENT ACROSS ALL SPATIAL INPUTS
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
    ], additional_targets={
        "topo": "image",
        "d2m":  "image",
    })

_AUG_PIPELINE = _make_aug_pipeline()


def augment_sample(fp_t, topo_t, d2m_t, aug_prob=0.5):
    def _rederive_coarse(fp):
        return F.avg_pool2d(fp.unsqueeze(0),
                            kernel_size=DS_FACTOR, stride=DS_FACTOR).squeeze(0)

    if _AUG_PIPELINE is None or np.random.rand() > aug_prob:
        return fp_t, topo_t, _rederive_coarse(fp_t), d2m_t

    fp_np   = fp_t.squeeze(0).numpy().astype(np.float32)
    topo_np = topo_t.squeeze(0).numpy().astype(np.float32)
    d2m_np  = (d2m_t.squeeze(0).numpy().astype(np.float32)
               if d2m_t is not None else np.zeros_like(fp_np))

    result   = _AUG_PIPELINE(image=fp_np, topo=topo_np, d2m=d2m_np)
    fp_aug   = torch.from_numpy(result["image"]).unsqueeze(0)
    topo_aug = torch.from_numpy(result["topo"]).unsqueeze(0)
    d2m_aug  = (torch.from_numpy(result["d2m"]).unsqueeze(0)
                if d2m_t is not None else None)
    coarse_aug = _rederive_coarse(fp_aug)
    return fp_aug, topo_aug, coarse_aug, d2m_aug


def augment_batch_consistent(fp, topo_1ch, xi_raw, d2m, dev):
    if not HAS_ALB:
        return fp, topo_1ch, xi_raw, d2m

    B       = fp.shape[0]
    fp_cpu  = fp.cpu(); tp_cpu = topo_1ch.cpu()
    d2m_cpu = d2m.cpu() if d2m is not None else None

    fp_list = []; tp_list = []; co_list = []; d2m_list = []
    for i in range(B):
        d2m_i = d2m_cpu[i] if d2m_cpu is not None else None
        fa, ta, ca, da = augment_sample(fp_cpu[i], tp_cpu[i], d2m_i)
        fp_list.append(fa); tp_list.append(ta); co_list.append(ca)
        if da is not None:
            d2m_list.append(da)

    fp_out  = torch.stack(fp_list).to(dev)
    tp_out  = torch.stack(tp_list).to(dev)
    co_out  = torch.stack(co_list).to(dev)
    d2m_out = torch.stack(d2m_list).to(dev) if d2m_list else d2m
    return fp_out, tp_out, co_out, d2m_out


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TOPO HELPERS
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


# ══════════════════════════════════════════════════════════════════════════════
# 7.  TEMPORAL CONDITIONING
# ══════════════════════════════════════════════════════════════════════════════
def build_temporal_cond(batch, dev, n_frames=T_COND):
    if "tc_frames" in batch:
        tc = batch["tc_frames"].to(dev)
        if tc.shape[1] >= n_frames:
            return tc[:, :n_frames]
        pad = torch.zeros(tc.shape[0], n_frames-tc.shape[1], *tc.shape[2:], device=dev)
        return torch.cat([tc, pad], dim=1)
    coarse    = batch["coarse"].to(dev)
    coarse_up = F.interpolate(coarse, scale_factor=4, mode='bilinear', align_corners=False)
    return coarse_up.expand(-1, n_frames, -1, -1)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  AUXILIARY LOSSES
# ══════════════════════════════════════════════════════════════════════════════
def spectral_loss(pred, target):
    P=torch.fft.rfft2(pred.float()); T=torch.fft.rfft2(target.float()); eps=1e-8
    return ((torch.log(P.abs()+eps)-torch.log(T.abs()+eps))**2).mean()

def hybrid_sigma_loss(pred, target, sigma, alpha=0.8):
    """
    Replaces EDM λ-MSE in corrdiff_residual mode.
    """
    eps = 1e-6
    # FORCE FLOAT32 to prevent FP16 overflow
    p = pred.float()
    t = target.float()
    mae = torch.abs(p - t)

    scaled_mae = (mae / (sigma + eps)).mean()

    w = (torch.sigmoid((t - 0.015) * 150) * 3.5 +
         torch.sigmoid((t - 0.08)  *  50) * 5.0 +
         torch.sigmoid((t - 0.25)  *  20) * 6.0 +
         torch.sigmoid((t - 0.5)   *  10) * 7.0 +
         (1.0 - torch.sigmoid((t - 0.015) * 150)) * 0.6)

    weighted_mae = (w * mae).mean()
    return alpha * scaled_mae + (1 - alpha) * weighted_mae

def pcc_loss(pred, target):
    B = pred.shape[0]
    # FORCE FLOAT32 to prevent FP16 sum() from exceeding 65,504
    p = pred.view(B, -1).float()
    t = target.view(B, -1).float()
    
    pm = p - p.mean(dim=1, keepdim=True)
    tm = t - t.mean(dim=1, keepdim=True)
    
    p_l2 = torch.sqrt((pm**2).sum(dim=1))
    t_l2 = torch.sqrt((tm**2).sum(dim=1))
    
    return (1 - (pm * tm).sum(dim=1) / (p_l2 * t_l2 + 1e-8)).mean()

@torch.no_grad()
def weighted_pcc(pred, target, lat_w=None):
    B = pred.shape[0]
    p = pred.float().view(B, -1); t = target.float().view(B, -1)
    if lat_w is not None:
        w = lat_w.view(1, -1).expand_as(p)
        w = w / w.sum(dim=1, keepdim=True)
        pm = (p * w).sum(dim=1, keepdim=True); tm = (t * w).sum(dim=1, keepdim=True)
        pm_c = p - pm; tm_c = t - tm
        r = (pm_c * tm_c * w).sum(dim=1) / (
            torch.sqrt((pm_c**2 * w).sum(dim=1) * (tm_c**2 * w).sum(dim=1)) + 1e-8)
    else:
        pm = p.mean(dim=1, keepdim=True); tm = t.mean(dim=1, keepdim=True)
        pm_c = p - pm; tm_c = t - tm
        p_l2 = torch.sqrt((pm_c**2).sum(dim=1))
        t_l2 = torch.sqrt((tm_c**2).sum(dim=1))
        r = (pm_c * tm_c).sum(dim=1) / (p_l2 * t_l2 + 1e-8)
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
    """
    Fractions Skill Score (FSS) for spatial verification.
    1.0 = Perfect forecast, 0.0 = No skill.
    """
    # Binarize based on rain threshold
    p_bin = (pred > threshold).float()
    t_bin = (target > threshold).float()
    
    # Compute fractions within the neighborhood window
    p_frac = F.avg_pool2d(p_bin, kernel_size=window, stride=1, padding=window//2)
    t_frac = F.avg_pool2d(t_bin, kernel_size=window, stride=1, padding=window//2)
    
    # FSS Formula
    mse = ((p_frac - t_frac)**2).mean(dim=[-1, -2])
    ref = (p_frac**2).mean(dim=[-1, -2]) + (t_frac**2).mean(dim=[-1, -2])
    
    fss = 1.0 - mse / (ref + 1e-8)
    return fss.mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# 10.  DDP
# ══════════════════════════════════════════════════════════════════════════════
def setup():
    rank=int(os.environ.get("RANK",0)); ws=int(os.environ.get("WORLD_SIZE",1))
    lr_=int(os.environ.get("LOCAL_RANK",0))
    if ws>1: dist.init_process_group("nccl"); torch.cuda.set_device(lr_)
    return rank,ws,lr_,torch.device(f"cuda:{lr_}" if torch.cuda.is_available() else "cpu")

def ar(t, ws):
    if ws>1 and dist.is_initialized(): dist.all_reduce(t,op=dist.ReduceOp.SUM); t/=ws
    return t


# ══════════════════════════════════════════════════════════════════════════════
# 11.  REGRESSOR LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_regressor(ckpt_path, dev):
    ck=torch.load(ckpt_path, map_location=dev)
    reg=CorrDiffRegressor(
        in_channels=ck.get("reg_in_channels",REG_IN_CH), out_channels=1,
        base_channels=64, channel_mult=(1,2,4), num_blocks=2,
        global_dim=GLOBAL_DIM, topo_channels=TOPO_CH,
        d2m_channels=ck.get("d2m_channels",D2M_CH), use_d2m=ck.get("use_d2m",True),
    ).to(dev)
    state={k.replace("module.",""):v for k,v in ck["model_state_dict"].items()}
    reg.load_state_dict(state); reg.eval()
    for p in reg.parameters(): p.requires_grad_(False)
    return reg


# ══════════════════════════════════════════════════════════════════════════════
# 12.  DDIM SAMPLER  (V2)
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def ddim_sample(raw_model, mu, tc, tp, gf, d2m, edm_schedule, dev):
    B = mu.shape[0]
    x_t = torch.randn_like(mu) * SIGMA_DATA
    sigmas = edm_schedule.to(dev)

    for i, sigma_cur in enumerate(sigmas):
        s_cur  = sigma_cur.view(1, 1, 1, 1)
        c_in   = 1. / torch.sqrt(s_cur**2 + SIGMA_DATA**2)
        c_out  = s_cur * SIGMA_DATA / torch.sqrt(s_cur**2 + SIGMA_DATA**2)
        c_skip = SIGMA_DATA**2 / (s_cur**2 + SIGMA_DATA**2)
        c_n    = (sigma_cur.log() / 4).expand(B)
        x_in   = torch.cat([x_t, mu, tc], dim=1)
        D_pred = raw_model(c_in * x_in, c_n, topo=tp,
                           global_features=gf, d2m=d2m, T=T_COND)
        x0_hat = c_skip * x_t[:, :1] + c_out * D_pred
        if i < len(sigmas) - 1:
            sigma_next = sigmas[i + 1].view(1, 1, 1, 1)
            x_t = x0_hat + sigma_next * (x_t - x0_hat) / s_cur.clamp(min=1e-8)
        else:
            x_t = x0_hat
    return x_t


# ══════════════════════════════════════════════════════════════════════════════
# 13.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train():
    rank,ws,lr_,dev=setup()
    os.makedirs(CKPT_DIR, exist_ok=True)

    SAVE    =os.path.join(CKPT_DIR,f"unet_{TRAIN_MODE}_sigma{SIGMA_DATA:.3f}_best.pth")
    LATEST  =os.path.join(CKPT_DIR,f"unet_{TRAIN_MODE}_sigma{SIGMA_DATA:.3f}_latest.pth")
    QDM_PATH=os.path.join(CKPT_DIR,f"qdm_{TRAIN_MODE}.pth")

    edm_schedule=_EDM_CPU.clone()

    ds =UpscaleDataset(RF_PATH,ORO_PATH,d2m_file=D2M_PATH,
                       split="train",normalize=True,device="cpu")
    n  =len(ds); trn=int(0.70*n); van=int(0.10*n)
    trl=DataLoader(Subset(ds,range(0,trn)),BATCH,
                   shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
    val=DataLoader(Subset(ds,range(trn,trn+van)),BATCH,
                   shuffle=False,num_workers=8,pin_memory=True)

    reg=load_regressor(REG_CKPT,dev)
    if rank==0: print(f"[Stage-1] Regressor loaded (frozen)")

    if TRAIN_MODE=="flow_matching":
        fm=FlowMatching(n_steps=FM_STEPS,cfg_scale=CFG_SCALE); loss_label="L_flow  "
    else:
        fm=None; loss_label="L_denoise"

    model=UNet(
        in_channels=UNET_IN_CH,out_channels=1,base_channels=BASE_CH,
        channel_mult=CHANNEL_MULT,num_res_blocks=NRB,dropout=DROPOUT,
        global_dim=GLOBAL_DIM,use_bottleneck_attention=True,
        topo_channels=TOPO_CH,use_d2m=True,d2m_channels=D2M_CH,
        temporal_frames=T_COND,
    ).to(dev)

    if ws>1:
        model=nn.parallel.DistributedDataParallel(
            model,device_ids=[lr_],find_unused_parameters=False)
    raw=model.module if ws>1 else model

    ema   =EMA(raw,decay=EMA_DECAY)
    opt   =AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY,betas=(0.9,0.999))
    scaler=GradScaler(device=dev.type)
    sched =CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=1,eta_min=MIN_LR)

    start=0; best_wpcc=0.0; no_improve=0

    if os.path.exists(LATEST):
        ck=torch.load(LATEST,map_location=dev)
        try:
            raw.load_state_dict({k.replace("module.",""):v
                                 for k,v in ck["model_state_dict"].items()})
        except RuntimeError as e:
            if rank==0: print(f"[RESUME WARNING] Partial load: {e}")
        opt.load_state_dict(ck["optimizer_state_dict"])
        start=ck.get("epoch",0)+1; best_wpcc=ck.get("wpcc",0.0)
        no_improve=ck.get("no_improve",0)
        if "ema_shadow" in ck:
            ema.shadow={k:v.to(dev) for k,v in ck["ema_shadow"].items()}
        if rank==0: print(f"[RESUME] ep={start}  best_wPCC={best_wpcc:.4f}")

    if rank==0:
        np_=sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL]  UNet {np_/1e6:.2f}M  mode={TRAIN_MODE}  sigma_data={SIGMA_DATA}")
        aug_info=("ON  fp+topo+d2m same geom tx | coarse re-derived from aug fp"
                  if HAS_ALB else "OFF (pip install albumentations)")
        print(f"[AUG]    {aug_info}")
        print(f"[LOSS]   corrdiff_residual → hybrid_sigma_loss(alpha=0.8)  λ-MSE removed")
        hdr=(f"{'Ep':>5}|{loss_label:>10}|{'L_spec':>8}|{'L_pcc':>8}"
             f"|{'Val':>8}|{'wPCC':>7}|{'CRPS':>8}|{'PSD_r':>7}|{'LR':>9}")
        print(hdr); print("-"*len(hdr))

    lat_w=None

    for ep in range(start, EPOCHS):
        model.train()
        t0=time.time(); sum_ml=sum_sp=sum_pc=nb=0.
        opt.zero_grad(set_to_none=True)

        for step, b in enumerate(trl, 1):
            try:
                fp       = b["fine"].to(dev)[:,PRECIP_CH:PRECIP_CH+1]
                topo_1ch = b["topo"].to(dev)
                xi_raw   = b["coarse"].to(dev)
                d2m      = b["d2m"].to(dev) if "d2m" in b else None
                var_map  = b["var_map"].to(dev)
                gf       = torch.stack([b["doy"],b["hour"]],1).float().to(dev)
                tc       = build_temporal_cond(b, dev)

                # FAST GPU-NATIVE AUGMENTATION
                if torch.rand(1).item() < 0.5:
                    fp = fp.flip(-1)
                    topo_1ch = topo_1ch.flip(-1)
                    if d2m is not None: d2m = d2m.flip(-1)
                if torch.rand(1).item() < 0.5:
                    fp = fp.flip(-2)
                    topo_1ch = topo_1ch.flip(-2)
                    if d2m is not None: d2m = d2m.flip(-2)
                
                # Re-derive coarse consistently
                xi_raw = F.avg_pool2d(fp, kernel_size=DS_FACTOR, stride=DS_FACTOR)

                tp  = expand_topo(topo_1ch)
                xi  = build_coarse_input(xi_raw, var_map)

                with torch.no_grad():
                    mu = reg(xi, topo=tp, global_features=gf, d2m=d2m)

                residual = fp - mu
                cfg_drop = (torch.rand(fp.shape[0], device=dev) < P_CFG_DROP)

                # ─────────────── V1: FLOW MATCHING ───────────────────────
                if TRAIN_MODE == "flow_matching":
                    x_t, t_vec, v_star = fm.get_train_sample(residual)
                    t_bcast = t_vec.view(-1,1,1,1)
                    x_in    = torch.cat([x_t, mu, tc], dim=1)
                    with autocast(device_type=dev.type):
                        v_pred   = model(x_in, t_vec, topo=tp, global_features=gf,
                                         cfg_drop=cfg_drop, d2m=d2m, T=T_COND)
                        l_main   = fm.loss(v_pred, v_star)
                        fine_hat = x_t + (1-t_bcast)*v_pred + mu

                # ─────────────── V2: CORRDIFF RESIDUAL ───────────────────
                else:
                    idx     = torch.randint(0, len(edm_schedule), (fp.shape[0],))
                    sigma_t = edm_schedule[idx].to(dev).view(-1,1,1,1)
                    eps     = torch.randn_like(residual)
                    x_t     = residual + sigma_t*eps
                    c_in    = 1./torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                    c_out   = sigma_t*SIGMA_DATA/torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                    c_skip  = SIGMA_DATA**2/(sigma_t**2 + SIGMA_DATA**2)
                    c_n     = (sigma_t.log()/4).view(fp.shape[0])
                    x_in    = torch.cat([x_t, mu, tc], dim=1)
                    with autocast(device_type=dev.type):
                        D_pred  = model(c_in*x_in, c_n, topo=tp, global_features=gf,
                                        cfg_drop=cfg_drop, d2m=d2m, T=T_COND)
                        x0_pred = c_skip*x_t[:,:1] + c_out*D_pred
                        # Hybrid sigma loss — replaces EDM λ-MSE
                        l_main  = hybrid_sigma_loss(x0_pred, residual, sigma_t, alpha=0.8)
                        fine_hat= x0_pred + mu

                # ── Shared auxiliary losses ───────────────────────────────
                with autocast(device_type=dev.type):
                    l_spec = spectral_loss(fine_hat, fp)
                    l_pcc  = pcc_loss(fine_hat, fp)
                    loss   = (l_main + LAMBDA_SPEC*l_spec + LAMBDA_PCC*l_pcc)/ACCUM_STEPS

                scaler.scale(loss).backward()

                if step % ACCUM_STEPS == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)
                    if step % 10 == 0:
                        ema.update(raw)

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
        lr_now=opt.param_groups[0]["lr"]

        # ── VALIDATION (EMA weights) ──────────────────────────────────
        live_state=deepcopy(raw.state_dict()); ema.apply_to(raw); model.eval()
        v_loss=torch.tensor(0.,device=dev); vn=torch.tensor(0,device=dev,dtype=torch.long)
        pcc_sum=crps_sum=psd_sum=fss_sum=0.; vb=0

        with torch.no_grad():
            for b in val:
                fp      = b["fine"].to(dev)[:,PRECIP_CH:PRECIP_CH+1]
                tp      = expand_topo(b["topo"].to(dev))
                gf      = torch.stack([b["doy"],b["hour"]],1).float().to(dev)
                xi_raw  = b["coarse"].to(dev)
                var_map = b["var_map"].to(dev)
                d2m     = b["d2m"].to(dev) if "d2m" in b else None
                tc      = build_temporal_cond(b, dev)
                xi      = build_coarse_input(xi_raw, var_map)

                with autocast(device_type=dev.type):
                    mu = reg(xi, topo=tp, global_features=gf, d2m=d2m)

                residual=fp-mu

                if TRAIN_MODE=="flow_matching":
                    x_t,t_vec,v_star=fm.get_train_sample(residual)
                    x_in=torch.cat([x_t,mu,tc],dim=1)
                    with autocast(device_type=dev.type):
                        v_pred=model(x_in,t_vec,topo=tp,global_features=gf,d2m=d2m,T=T_COND)
                        l_val=fm.loss(v_pred,v_star)
                else:
                    idx=torch.randint(0,len(edm_schedule),(fp.shape[0],))
                    sigma_t=edm_schedule[idx].to(dev).view(-1,1,1,1)
                    eps=torch.randn_like(residual); x_t=residual+sigma_t*eps
                    c_in=1./torch.sqrt(sigma_t**2+SIGMA_DATA**2)
                    c_out=sigma_t*SIGMA_DATA/torch.sqrt(sigma_t**2+SIGMA_DATA**2)
                    c_skip=SIGMA_DATA**2/(sigma_t**2+SIGMA_DATA**2)
                    c_n=(sigma_t.log()/4).view(fp.shape[0])
                    x_in=torch.cat([x_t,mu,tc],dim=1)
                    with autocast(device_type=dev.type):
                        D_pred=model(c_in*x_in,c_n,topo=tp,global_features=gf,d2m=d2m,T=T_COND)
                        x0_pred=c_skip*x_t[:,:1]+c_out*D_pred
                        # Hybrid sigma loss — replaces EDM λ-MSE in validation too
                        l_val=hybrid_sigma_loss(x0_pred, residual, sigma_t, alpha=0.8)

                v_loss+=l_val*fp.shape[0]; vn+=fp.shape[0]

                samples=[]
                for _ in range(N_ENS):
                    if TRAIN_MODE=="flow_matching":
                        x_cond=torch.cat([mu,tc],dim=1)
                        s=fm.sample(raw,x_cond,topo=tp,global_features=gf,
                                    d2m=d2m,cfg_scale=CFG_SCALE,T=T_COND)+mu
                    else:
                        s=ddim_sample(raw,mu,tc,tp,gf,d2m,edm_schedule,dev)+mu
                    s=PhysicsGuide.apply(s,xi_raw,enforce_mass=True,enforce_dry=True)
                    samples.append(s)

                samples_t=torch.stack(samples); best_s=samples_t.mean(0)
                pcc_sum +=weighted_pcc(best_s,fp,lat_w)
                crps_sum+=crps_ensemble(samples_t,fp)
                psd_sum +=psd_tail_ratio(best_s,fp)
                fss_sum +=fractions_skill_score(best_s,fp, threshold=0.5, window=5) # <--- NEW
                vb+=1

        ema.restore(raw,live_state); model.train()

        if ws>1 and dist.is_initialized():
            dist.barrier()
            for t__ in [v_loss,vn]: dist.all_reduce(t__,op=dist.ReduceOp.SUM)
            dist.barrier()

        vw=(v_loss/vn.clamp(1)).item()
        wpcc=pcc_sum/max(vb,1); crps_v=crps_sum/max(vb,1); psd_r=psd_sum/max(vb,1)
        fss_v=fss_sum/max(vb,1)
        el=time.time()-t0

        if rank==0:
            star=" ★" if wpcc>best_wpcc else ""
            print(f"{ep:>5}|{te_ml:>10.5f}|{te_sp:>8.5f}|{te_pc:>8.5f}"
                  f"|{vw:>8.4f}|{wpcc:>7.4f}|{crps_v:>8.4f}|{psd_r:>7.3f}|{fss_v:>6.3f}"
                  f"|{lr_now:>9.2e}  [{el:.0f}s]{star}")

            ck_base={
                "epoch":ep,"model_state_dict":raw.state_dict(),
                "optimizer_state_dict":opt.state_dict(),"ema_shadow":ema.shadow,
                "val_loss":vw,"wpcc":wpcc,"crps":crps_v,"psd_tail_ratio":psd_r,
                "no_improve":no_improve,"train_mode":TRAIN_MODE,"sigma_data":SIGMA_DATA,
                "unet_in_channels":UNET_IN_CH,"t_cond":T_COND,"d2m_channels":D2M_CH,
            }
            torch.save(ck_base, LATEST)

            if wpcc>best_wpcc:
                best_wpcc=wpcc; no_improve=0
                torch.save(ck_base, SAVE)
                print(f"  ★ BEST  val_loss={vw:.4f}  wPCC={wpcc:.4f}  "
                      f"CRPS={crps_v:.4f}  PSD_r={psd_r:.3f}")
                if (ep+1)%50==0:
                    torch.save(ck_base, os.path.join(
                        CKPT_DIR,f"unet_{TRAIN_MODE}_ep{ep+1:04d}_wpcc{wpcc:.4f}.pth"))
            else:
                no_improve+=1

        if no_improve>=PATIENCE:
            if rank==0: print(f"\n⚠  Early stop ep={ep+1}  best_wPCC={best_wpcc:.4f}")
            break

    # ── QDM calibration ──────────────────────────────────────────────────────
    if rank==0:
        print("\n[QDM] fitting on val set …")
        qdm=QDM(n_quantiles=1000)
        ck_best=torch.load(SAVE,map_location=dev)
        raw.load_state_dict({k.replace("module.",""):v
                             for k,v in ck_best["model_state_dict"].items()})
        if "ema_shadow" in ck_best:
            ema.shadow={k:v.to(dev) for k,v in ck_best["ema_shadow"].items()}
            ema.apply_to(raw)
        model.eval(); all_pred,all_obs=[],[]
        with torch.no_grad():
            for b in val:
                fp     =b["fine"].to(dev)[:,PRECIP_CH:PRECIP_CH+1]
                tp     =expand_topo(b["topo"].to(dev))
                gf     =torch.stack([b["doy"],b["hour"]],1).float().to(dev)
                xi_raw =b["coarse"].to(dev); var_map=b["var_map"].to(dev)
                d2m    =b["d2m"].to(dev) if "d2m" in b else None
                tc     =build_temporal_cond(b,dev)
                xi     =build_coarse_input(xi_raw,var_map)
                mu     =reg(xi,topo=tp,global_features=gf,d2m=d2m)
                if TRAIN_MODE=="flow_matching":
                    x_cond=torch.cat([mu,tc],dim=1)
                    s=fm.sample(raw,x_cond,topo=tp,global_features=gf,
                                d2m=d2m,cfg_scale=CFG_SCALE,T=T_COND)+mu
                else:
                    s=ddim_sample(raw,mu,tc,tp,gf,d2m,edm_schedule,dev)+mu
                pred=PhysicsGuide.apply(s,xi_raw,enforce_mass=True,enforce_dry=True)
                all_pred.append(pred.cpu()); all_obs.append(fp.cpu())
        qdm.fit(torch.cat(all_pred),torch.cat(all_obs))
        qdm.save(QDM_PATH); print(f"[QDM] saved → {QDM_PATH}")

    if ws>1 and dist.is_initialized(): dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════════
# 14.  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
class CorrDiffInference:
    def __init__(self, reg_ckpt, unet_ckpt, qdm_ckpt, device,
                 cfg_scale=CFG_SCALE, n_ens=8):
        self.dev=device; self.cfg_scale=cfg_scale; self.n_ens=n_ens
        self.reg=load_regressor(reg_ckpt,device)
        ck=torch.load(unet_ckpt,map_location=device)
        tc=ck.get("t_cond",T_COND); self.tc=tc
        self.train_mode=ck.get("train_mode","flow_matching")
        self.fm=FlowMatching(n_steps=FM_STEPS,cfg_scale=cfg_scale)
        self.edm_sched=build_edm_schedule(FM_STEPS,sigma_data=ck.get("sigma_data",SIGMA_DATA))
        self.unet=UNet(
            in_channels=ck.get("unet_in_channels",UNET_IN_CH),out_channels=1,
            base_channels=BASE_CH,channel_mult=CHANNEL_MULT,num_res_blocks=NRB,dropout=0.,
            global_dim=GLOBAL_DIM,topo_channels=TOPO_CH,use_d2m=True,d2m_channels=D2M_CH,
            temporal_frames=tc,
        ).to(device)
        state=({k:v.to(device) for k,v in ck["ema_shadow"].items()} if "ema_shadow" in ck
               else {k.replace("module.",""):v for k,v in ck["model_state_dict"].items()})
        self.unet.load_state_dict(state); self.unet.eval()
        self.qdm=QDM.load(qdm_ckpt) if qdm_ckpt and os.path.exists(qdm_ckpt) else None

    @torch.no_grad()
    def predict(self, coarse, var_map, topo, d2m, doy, hour, tc_frames=None):
        dev=self.dev
        coarse=coarse.to(dev); var_map=var_map.to(dev)
        topo=topo.to(dev); d2m=d2m.to(dev)
        gf=torch.stack([doy.to(dev),hour.to(dev)],dim=1).float()
        xi=build_coarse_input(coarse,var_map); tp=expand_topo(topo)
        if tc_frames is None:
            coarse_up=F.interpolate(coarse,scale_factor=4,mode='bilinear',align_corners=False)
            tc_frames=coarse_up.expand(-1,self.tc,-1,-1).to(dev)
        else:
            tc_frames=tc_frames.to(dev)
        mu=self.reg(xi,topo=tp,global_features=gf,d2m=d2m)
        samples=[]
        for _ in range(self.n_ens):
            if self.train_mode=="flow_matching":
                x_cond=torch.cat([mu,tc_frames],dim=1)
                s=self.fm.sample(self.unet,x_cond,topo=tp,global_features=gf,
                                 d2m=d2m,cfg_scale=self.cfg_scale,T=self.tc)+mu
            else:
                s=ddim_sample(self.unet,mu,tc_frames,tp,gf,d2m,self.edm_sched,dev)+mu
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
    parser.add_argument("--mode",    default=None,choices=["flow_matching","corrdiff_residual"])
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--patience",type=int,   default=None)
    args=parser.parse_args()
    if args.mode     is not None and "TRAIN_MODE" not in os.environ: TRAIN_MODE=args.mode
    if args.epochs   is not None: EPOCHS=args.epochs
    if args.batch    is not None: BATCH=args.batch
    if args.lr       is not None: LR=args.lr
    if args.dropout  is not None: DROPOUT=args.dropout
    if args.patience is not None: PATIENCE=args.patience
    train()
