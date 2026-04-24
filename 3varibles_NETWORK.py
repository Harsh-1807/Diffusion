# -*- coding: utf-8 -*-
"""
CorrDiff Network — Stable v7
============================
+ D2M (dewpoint 2m) as 4th auxiliary spatial input
  → Regressor: extra stem channel
  → UNet: concatenated into x before head
+ CoordConv2d, FourierFilter, Topo-FiLM ResBlock, CFG support
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
# REGRESSOR  (Stage 1)
# topo : [B, 3, H, W]  — elevation + slope + aspect
# d2m  : [B, 1, H, W]  — dewpoint z-scored (optional)
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

        # d2m gets its own lightweight stem, adds to r features
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

        # add d2m into rain branch (same spatial res after upsample)
        if self.use_d2m and d2m is not None:
            # d2m is at fine res [B,1,H,W]; r is already at fine res
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
# UNET  (Stage 2) — CFG-aware
# topo     : [B, 3, H, W]
# d2m      : [B, 1, H, W]  — concatenated into x before head
# cfg_drop : bool [B]
# ================================================================

class ResBlock(nn.Module):
    def __init__(self, ic, oc, ec, down=False, up=False,
                 use_topo=True, topo_channels=3):
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
        return self.se(self.c2(F.silu(h_norm))) + self.sk(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=128,
                 channel_mult=(1,2,2,4), num_res_blocks=2, num_blocks=None,
                 global_dim=2, use_bottleneck_attention=True,
                 topo_channels=3, use_d2m=True, **kw):
        super().__init__()
        nrb = num_blocks if num_blocks else num_res_blocks
        ec  = base_channels * 4
        self.topo_channels = topo_channels
        self.use_d2m = use_d2m
        # d2m adds 1 extra channel to UNet input
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
                self.downs.append(ResBlock(ch, oc, ec, topo_channels=topo_channels))
                ch = oc; sk.append(ch)
            self.downs.append(ResBlock(ch, ch, ec, down=True, topo_channels=topo_channels))
            sk.append(ch)

        self.m1  = ResBlock(ch, ch, ec, topo_channels=topo_channels)
        self.ma  = BnAttn(ch, max(1, ch//64)) if use_bottleneck_attention else nn.Identity()
        self.fft = FourierFilter(ch)
        self.m2  = ResBlock(ch, ch, ec, topo_channels=topo_channels)

        for m in reversed(channel_mult):
            oc = base_channels * m
            self.ups.append(ResBlock(ch+sk.pop(), oc, ec, up=True, topo_channels=topo_channels))
            ch = oc
            for _ in range(nrb):
                self.ups.append(ResBlock(ch+sk.pop(), oc, ec, topo_channels=topo_channels))
                ch = oc

        self.out = nn.Sequential(
            nn.GroupNorm(_g(ch), ch), nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1))

    @staticmethod
    def _sin_emb(sigma, dim):
        half = dim // 2
        freq = torch.exp(torch.arange(half, device=sigma.device) * (-math.log(10000)/(half-1)))
        t = (sigma.clamp(1e-4).log() * 0.25).unsqueeze(1)
        e = t * freq.unsqueeze(0)
        return torch.cat([e.sin(), e.cos()], -1)

    def forward(self, x, t, topo=None, global_features=None, cfg_drop=None, d2m=None):
        """
        x        : [B, C, H, W]  noisy input
        d2m      : [B, 1, H, W]  z-scored dewpoint (optional)
        cfg_drop : bool [B]
        """
        raw = self._sin_emb(t, self.t_emb[0].in_features)
        emb = self.t_emb(raw)

        if global_features is not None and self.g_mlp:
            gf = global_features.clone()
            if cfg_drop is not None: gf[cfg_drop] = 0.
            emb = emb + self.g_mlp(gf)

        x_in    = x.clone()
        topo_in = topo.clone() if topo is not None else None

        if cfg_drop is not None and cfg_drop.any():
            x_in[cfg_drop, 1:] = 0.
            if topo_in is not None: topo_in[cfg_drop] = 0.

        # concat d2m as extra spatial channel
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
        h = self.m2(h, emb, topo_in)
        for l in self.ups: h = torch.cat([h, sk.pop()], 1); h = l(h, emb, topo_in)
        return self.out(h)


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
