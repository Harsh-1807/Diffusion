# -*- coding: utf-8 -*-
"""Stage-2: Train CorrDiff Diffusion — with D2M support
Based on stable v7 with:
  + D2M (dewpoint 2m) passed through regressor and UNet
  + CFG training: 10% random condition dropout
  + CFG sampling: guidance_scale=1.5
  + Topo expanded to 3ch (elevation, slope, aspect) via Sobel
  + Spectral loss in physical space
"""

import os, json, time, traceback
import torch, torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Dataset import UpscaleDataset
from Network import UNet, CorrDiffRegressor, QDM

RF_PATH  = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
ORO_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"
D2M_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/era5_aligned_to_rf.nc"
REG_PATH = "checkpoints/regressor/regressor_best.pth"
CKPT_DIR = "checkpoints/diffusion/2/"
SAVE     = "checkpoints/diffusion/2/diffusion_best.pth"
LATEST   = "checkpoints/diffusion/2/diffusion_latest.pth"
QDM_PATH = "checkpoints/diffusion/2/qdm.pth"

BATCH         = 16
LR            = 1e-4
EPOCHS        = 2000
PRECIP        = 0
GAMMA         = 1.0
P_MEAN        = -1.2
P_STD         = 1.2
T_COND        = 5
# noisy(1) + mu(1) + temporal(T_COND) = 7
# d2m(1) concatenated inside UNet.forward via use_d2m=True — NOT counted here
UNET_IN       = 2 + T_COND        # = 7
WET           = 0.1
CFG_DROP_PROB = 0.10
CFG_GUIDANCE  = 1.5
CKPT_EVERY    = 100
MAX_CKPT      = 16
QDM_BATCHES   = 50
QDM_N         = 2000


# ── Sobel topo expansion ────────────────────────────────────────

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
    return torch.cat([compute_slope_aspect(topo_1ch[i:i+1]) for i in range(topo_1ch.shape[0])], dim=0)


# ── DDP ─────────────────────────────────────────────────────────

def setup():
    rank = int(os.environ.get("RANK", 0)); ws = int(os.environ.get("WORLD_SIZE", 1))
    lr_  = int(os.environ.get("LOCAL_RANK", 0))
    if ws > 1: dist.init_process_group("nccl"); torch.cuda.set_device(lr_)
    return rank, ws, lr_, torch.device(f"cuda:{lr_}" if torch.cuda.is_available() else "cpu")

def ar(t, ws):
    if ws > 1 and dist.is_initialized(): dist.all_reduce(t, op=dist.ReduceOp.SUM); t /= ws
    return t


# ── Math helpers ─────────────────────────────────────────────────

def ag(x): return torch.sign(x)*torch.pow(torch.abs(x)+1e-8, 1./GAMMA)
def ig(x): return torch.sign(x)*torch.pow(torch.abs(x)+1e-8, GAMMA)

def edm_p(sig, sd):
    s2, sd2 = sig**2, sd**2; d = s2 + sd2
    return sd2/d, sig*sd/d.sqrt(), 1./d.sqrt(), 0.25*sig.log()

def edm_l(pred, tgt, sig, sd):
    w = ((sig**2 + sd**2)/(sig*sd)**2).clamp(.1, 10.)
    return (w.view(-1,1,1,1)*(pred - tgt)**2).mean()

def pcc(p, t):
    p = p.flatten(1); t = t.flatten(1)
    vp = p - p.mean(1, keepdim=True); vt = t - t.mean(1, keepdim=True)
    return (vp*vt).sum(1) / (vp.pow(2).sum(1).sqrt()*vt.pow(2).sum(1).sqrt()).clamp(1e-6)

def wpcc(p, o):
    wet = o.flatten(1).max(1).values > WET
    if wet.sum() < 5: return p.new_tensor(0.)
    return pcc(p[wet], o[wet]).clamp(-1, 1).mean()

def spectral_loss(p, t):
    Pm = torch.abs(torch.fft.rfft2(p.float())) + 1e-8
    Tm = torch.abs(torch.fft.rfft2(t.float())) + 1e-8
    return F.mse_loss(torch.log1p(Pm), torch.log1p(Tm))

def dhr(p, o):
    dry = o.flatten(1).max(1).values < WET
    return (p[dry].flatten(1).max(1).values < WET).float().mean() if dry.any() else p.new_tensor(1.)

def hrmse(p, o):
    w = o[o > WET]
    if w.numel() < 10: return p.new_tensor(0.)
    thr = torch.quantile(w, .9)
    hv  = o.flatten(1).max(1).values > thr
    return F.mse_loss(p[hv], o[hv]).sqrt() if hv.any() else p.new_tensor(0.)


# ── EMA ──────────────────────────────────────────────────────────

class EMA:
    def __init__(self, m, d=0.9995, wu=1000):
        self.d = d; self.wu = wu; self.step = 0
        self.s = {n:p.data.clone() for n,p in m.named_parameters() if p.requires_grad}
        self._b = None
    def _decay(self): return 0.99 + min(self.step/max(self.wu,1),1.)*(self.d-0.99)
    def update(self, m):
        self.step += 1; d = self._decay()
        with torch.no_grad():
            for n,p in m.named_parameters():
                if p.requires_grad: self.s[n].mul_(d).add_(p.data, alpha=1-d)
    def apply(self, m):
        for n,p in m.named_parameters():
            if p.requires_grad and n in self.s: p.data.copy_(self.s[n])
    def backup(self, m): self._b = {n:p.data.clone() for n,p in m.named_parameters() if p.requires_grad}
    def restore(self, m):
        if not self._b: return
        for n,p in m.named_parameters():
            if p.requires_grad and n in self._b: p.data.copy_(self._b[n])
        self._b = None


# ── Temporal conditioning ────────────────────────────────────────

def build_tc(ds, idx, fs, dev):
    c = ds.coarse; frames = []
    for k in range(T_COND, 0, -1):
        fk = torch.stack([c[max(0, int(i)-k)] for i in idx]).to(dev)
        fk = F.interpolate(fk.float(), (fs, fs), mode='bilinear', align_corners=False)
        frames.append(fk)
    return torch.cat(frames, 1)


# ── Sampling (CFG) ───────────────────────────────────────────────

@torch.no_grad()
def sample(model, reg, xi, topo_3ch, tf, dsd, ds, idx,
           n_steps=20, guidance_scale=CFG_GUIDANCE, qdm=None, d2m=None):
    model.eval()
    m = model.module if hasattr(model, "module") else model

    # regressor forward — pass d2m
    mu = F.softplus(reg(xi, topo=topo_3ch, global_features=tf, d2m=d2m), beta=5)

    H_t, W_t = topo_3ch.shape[2], topo_3ch.shape[3]
    tc = build_tc(ds, idx, H_t, xi.device) if idx else \
         torch.zeros(xi.shape[0], T_COND, H_t, W_t, device=xi.device)

    smax  = 3.*dsd; smin = .002; rho = 9.
    steps = torch.arange(n_steps+1, device=xi.device, dtype=torch.float32)/n_steps
    sigs  = (smax**(1/rho) + steps*(smin**(1/rho)-smax**(1/rho)))**rho
    x     = torch.randn_like(mu)*sigs[0]

    def dn_cond(xn, sig, drop=False):
        sb = sig.expand(mu.shape[0])
        cs, co, ci, cn = edm_p(sb, dsd)
        ni  = torch.cat([ci.view(-1,1,1,1)*xn, mu, tc], 1).clamp(-10, 10)
        cfg = torch.ones(mu.shape[0], dtype=torch.bool, device=xi.device) if drop else None
        # UNet gets d2m — concatenated internally via use_d2m=True
        Fth = m(ni, cn, topo=topo_3ch, global_features=tf, cfg_drop=cfg, d2m=d2m)
        return cs.view(-1,1,1,1)*xn + co.view(-1,1,1,1)*Fth

    def dn_cfg(xn, sig):
        cond   = dn_cond(xn, sig, drop=False)
        uncond = dn_cond(xn, sig, drop=True)
        return uncond + guidance_scale*(cond - uncond)

    for i in range(n_steps):
        s, sn = sigs[i], sigs[i+1]
        gamma = min(0.2, 2.**0.5-1.) if s > 0.01 else 0.
        s_hat = s*(1+gamma)
        if gamma > 0: x = x + torch.randn_like(x)*torch.sqrt(s_hat**2 - s**2)

        Dx = dn_cfg(x, s_hat)
        d  = (x - Dx)/s_hat
        if i == n_steps-1:
            x = x + (sn - s_hat)*d
        else:
            xn_  = x + (sn - s_hat)*d
            Dx2  = dn_cfg(xn_, sn)
            d2_  = (xn_ - Dx2)/sn
            x    = x + (sn - s_hat)*(d + d2_)*0.5

    phys = torch.expm1((mu + x*dsd).clamp(0, 5.0))
    if qdm is not None and qdm._fitted:
        c = qdm.apply(phys); c[phys < WET] = 0.; return c
    return phys


@torch.no_grad()
def sample_K(model, reg, xi, tp3, tf, dsd, ds, idx, K=4, n=20, qdm=None, d2m=None):
    model.eval()
    return torch.stack([sample(model, reg, xi, tp3, tf, dsd, ds, idx, n, qdm=qdm, d2m=d2m)
                        for _ in range(K)]).mean(0)


@torch.no_grad()
def fit_qdm(model, reg, loader, dsd, ds, dev):
    model.eval(); ap = []; ao = []
    for bi, b in enumerate(loader):
        if bi >= QDM_BATCHES: break
        fv   = b["fine"].to(dev)
        tv3  = expand_topo(b["topo"].to(dev))
        tf   = torch.stack([b["doy"],b["hour"]],1).float().to(dev)
        xi   = b["coarse"].to(dev)
        d2m  = b["d2m"].to(dev) if "d2m" in b else None
        idx  = b["idx"].tolist() if "idx" in b else None
        ap.append(sample(model, reg, xi, tv3, tf, dsd, ds, idx, 20, d2m=d2m).cpu())
        ao.append(torch.expm1(fv[:,:1].clamp(0,5.0)).cpu())
    ap = torch.cat(ap).flatten(); ao = torch.cat(ao).flatten()
    wet = ao > WET
    print(f"  [QDM] wet={wet.float().mean():.3f} ({wet.sum():,})")
    if wet.sum() < 1000: return QDM(QDM_N)
    ap = ap[wet]; ao = ao[wet]
    if ap.numel() > 10_000_000:
        i = torch.randperm(ap.numel())[:10_000_000]; ap = ap[i]; ao = ao[i]
    q = QDM(QDM_N); q.fit(ap, ao); return q


def save_manifest(path, ckpt, wv, mk=MAX_CKPT):
    m = json.load(open(path)) if os.path.exists(path) else {"checkpoints":[]}
    m["checkpoints"].append({"path":ckpt,"wpcc":float(wv)})
    m["checkpoints"].sort(key=lambda x:x["wpcc"], reverse=True)
    for e in m["checkpoints"][mk:]:
        p = e["path"]
        if os.path.exists(p) and p not in (SAVE, LATEST):
            try: os.remove(p)
            except: pass
    m["checkpoints"] = m["checkpoints"][:mk]
    json.dump(m, open(path,"w"), indent=2)


# ── Training ─────────────────────────────────────────────────────

def train():
    rank, ws, lr_, dev = setup()
    os.makedirs(CKPT_DIR, exist_ok=True)

    # dataset — d2m_file passed
    ds  = UpscaleDataset(RF_PATH, ORO_PATH, d2m_file=D2M_PATH,
                         split="train", normalize=True, device="cpu")
    n   = len(ds); trn = int(.70*n); van = int(.10*n)
    trl = DataLoader(Subset(ds, range(0,trn)),       BATCH, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    val = DataLoader(Subset(ds, range(trn,trn+van)), BATCH, shuffle=False, num_workers=4, pin_memory=True)
    fs  = ds.H
    if rank == 0: print(f"[DATA] Train={trn} Val={van} Fine={fs}  UNET_IN={UNET_IN}  D2M=YES")

    # frozen regressor — use_d2m=True
    ck_r = torch.load(REG_PATH, map_location=dev)
    reg  = CorrDiffRegressor(in_channels=1, out_channels=1, base_channels=64,
                              channel_mult=(1,2,4), num_blocks=2, global_dim=2,
                              topo_channels=3, use_d2m=True).to(dev)
    sd_key = "ema_state_dict" if "ema_state_dict" in ck_r else "model_state_dict"
    miss, unex = reg.load_state_dict(ck_r[sd_key], strict=False)
    if rank == 0: print(f"[REG] {len(miss)} missing {len(unex)} unexpected")
    reg.eval()
    for p in reg.parameters(): p.requires_grad_(False)

    dsd = float(ck_r.get("sigma_data", 0.9020))
    if rank == 0: print(f"[DIFF_SD] {dsd:.4f}")

    # UNet — use_d2m=True; d2m concatenated inside UNet.forward (adds 1ch internally)
    model = UNet(
        in_channels=UNET_IN, out_channels=1, base_channels=96,
        channel_mult=(1,2,2,4), num_res_blocks=2, global_dim=2,
        use_bottleneck_attention=True, topo_channels=3, use_d2m=True
    ).to(dev)
    if ws > 1: model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_])
    raw = model.module if ws > 1 else model; ema = EMA(raw)

    opt    = AdamW(model.parameters(), lr=LR, weight_decay=1e-4, betas=(0.9,0.999))
    scaler = GradScaler(device=dev.type, init_scale=256, growth_interval=500, backoff_factor=0.5)
    sched  = CosineAnnealingWarmRestarts(opt, T_0=200, T_mult=2, eta_min=LR*.01)

    start = 0; best = -1.
    qdm   = QDM(QDM_N)
    if os.path.exists(QDM_PATH):
        try: qdm = QDM.load(QDM_PATH); print(f"[QDM] Loaded")
        except: pass

    for rp in [LATEST, SAVE]:
        if os.path.exists(rp):
            ck2 = torch.load(rp, map_location=dev)
            raw.load_state_dict(ck2["model_state_dict"])
            if "ema_state_dict" in ck2:
                ema.s = {k:v.clone().to(dev) for k,v in ck2["ema_state_dict"].items()}
            start = ck2.get("epoch",0)+1; best = ck2.get("val_wpcc",-1.)
            if rank == 0: print(f"[RESUME] ep={start} best={best:.4f}"); break

    if rank == 0:
        np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] UNet {np_/1e6:.2f}M  CFG_DROP={CFG_DROP_PROB}  GUIDE={CFG_GUIDANCE}")
        hdr = f"{'Ep':>6}|{'EDM':>7}|{'wPCC_tr':>8}|{'wPCC_v':>8}|{'DHR':>6}|{'HRMSE':>7}|QDM"
        print(hdr); print("-"*len(hdr))

    for ep in range(start, EPOCHS):
        model.train(); t0 = time.time(); se = sw = nb = 0.

        for b in trl:
            try:
                fp  = b["fine"].to(dev)[:,PRECIP:PRECIP+1]
                tp3 = expand_topo(b["topo"].to(dev))                # [B,3,H,W]
                tf  = torch.stack([b["doy"],b["hour"]],1).float().to(dev)
                xi  = b["coarse"].to(dev)
                d2m = b["d2m"].to(dev) if "d2m" in b else None      # [B,1,H,W]
                idx = b["idx"].tolist() if "idx" in b else None
                tc  = build_tc(ds, idx, fs, dev) if idx else \
                      torch.zeros(fp.shape[0], T_COND, fs, fs, device=dev)

                with torch.no_grad():
                    mu = F.softplus(reg(xi, topo=tp3, global_features=tf, d2m=d2m), beta=5)

                tr_raw = (fp - mu)/dsd; tr_tgt = ag(tr_raw)
                sig    = (torch.randn(fp.shape[0], device=dev)*P_STD+P_MEAN).exp().clamp(.01, 1.5*dsd)
                cs, co, ci, cn = edm_p(sig, dsd)
                nr  = tr_tgt + torch.randn_like(tr_tgt)*sig.view(-1,1,1,1)
                ni  = torch.cat([ci.view(-1,1,1,1)*nr, mu, tc], 1).clamp(-10, 10)

                cfg_drop = torch.rand(fp.shape[0], device=dev) < CFG_DROP_PROB

                opt.zero_grad(set_to_none=True)
                with autocast(device_type=dev.type):
                    # d2m and cfg_drop passed to UNet
                    Fth = model(ni, cn, topo=tp3, global_features=tf, cfg_drop=cfg_drop, d2m=d2m)
                    pr  = cs.view(-1,1,1,1)*nr + co.view(-1,1,1,1)*Fth

                    pf  = mu + ig(pr)*dsd
                    pfl = torch.expm1(pf.clamp(0, 5.0))
                    fpl = torch.expm1(fp.clamp(0, 5.0))

                    le    = edm_l(pr, tr_tgt, sig, dsd)
                    lpcc  = (1 - pcc(pr, tr_tgt)).mean()
                    lspec = spectral_loss(pfl, fpl)
                    loss  = le + 0.05*lpcc + 0.1*lspec

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                scaler.step(opt); scaler.update(); ema.update(raw)
                with torch.no_grad(): se += le.item(); sw += wpcc(pfl, fpl).item(); nb += 1
            except Exception:
                if rank == 0: traceback.print_exc(); raise

        sched.step(); ni = 1./max(nb,1)
        te = ar(torch.tensor(se*ni, device=dev), ws).item()
        tw = ar(torch.tensor(sw*ni, device=dev), ws).item()

        ema.backup(raw); ema.apply(raw); model.eval()
        vs = torch.tensor(0., device=dev); vd = torch.tensor(0., device=dev)
        vh = torch.tensor(0., device=dev); vn = torch.tensor(0, device=dev, dtype=torch.long)
        DO_VAL = (ep%5==0) or (ep==start)

        with torch.no_grad():
            if DO_VAL:
                for bi, b in enumerate(val):
                    if bi >= 10: break
                    fv    = b["fine"].to(dev)
                    tv3   = expand_topo(b["topo"].to(dev))
                    tf2   = torch.stack([b["doy"],b["hour"]],1).float().to(dev)
                    xi_v  = b["coarse"].to(dev)
                    d2m_v = b["d2m"].to(dev) if "d2m" in b else None
                    idx_v = b["idx"].tolist() if "idx" in b else None
                    fpl   = torch.expm1(fv[:,:1].clamp(0,5.0))
                    pfl   = sample_K(model, reg, xi_v, tv3, tf2, dsd, ds, idx_v,
                                     K=4, n=20, d2m=d2m_v)
                    vs += wpcc(pfl,fpl); vd += dhr(pfl,fpl); vh += hrmse(pfl,fpl); vn += 1

        if ws > 1 and dist.is_initialized():
            dist.barrier()
            for t__ in [vs,vd,vh,vn]: dist.all_reduce(t__, op=dist.ReduceOp.SUM)
            dist.barrier()

        if DO_VAL and vn > 0: vw=(vs/vn).item(); vdhr=(vd/vn).item(); vhrm=(vh/vn).item()
        else: vw=best; vdhr=0.; vhrm=0.

        qs = "✓" if qdm._fitted else "✗"
        ema.restore(raw); el = time.time()-t0

        if rank == 0 and ep%5==0:
            print(f"{ep:>6}|{te:>7.3f}|{tw:>8.4f}|{vw:>8.4f}|{vdhr:>6.3f}|{vhrm:>7.3f}|{qs} [{el:.0f}s]")

        if rank == 0:
            ck = {"epoch":ep, "model_state_dict":raw.state_dict(),
                  "ema_state_dict":{k:v.cpu() for k,v in ema.s.items()},
                  "optimizer_state_dict":opt.state_dict(),
                  "val_wpcc":vw, "diff_sd":dsd, "unet_in":UNET_IN}
            torch.save(ck, LATEST)

            if DO_VAL and vw > best:
                best = vw; ema.backup(raw); ema.apply(raw)
                b2 = dict(ck); b2["model_state_dict"] = raw.state_dict(); b2["ema_state_dict"] = raw.state_dict()
                torch.save(b2, SAVE); ema.restore(raw)
                print(f"  ★ BEST wPCC={vw:.4f}")
                print("  [QDM] Fitting..."); ema.backup(raw); ema.apply(raw)
                qdm = fit_qdm(model, reg, val, dsd, ds, dev)
                if qdm._fitted: qdm.save(QDM_PATH); print("  [QDM] Saved")
                ema.restore(raw)

            if DO_VAL and ep>0 and ep%CKPT_EVERY==0 and vw>0:
                ep_p = os.path.join(CKPT_DIR, f"diffusion_ep{ep:05d}.pth")
                ema.backup(raw); ema.apply(raw)
                ec = dict(ck); ec["model_state_dict"] = raw.state_dict(); ec["ema_state_dict"] = raw.state_dict()
                torch.save(ec, ep_p); ema.restore(raw)
                save_manifest(os.path.join(CKPT_DIR,"manifest.json"), ep_p, vw)

    if ws > 1 and dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__": train()
