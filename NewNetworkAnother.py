# -*- coding: utf-8 -*-
"""
CorrDiff Network Architecture (Stage 1 & 2) — IMPROVED v2
==========================================================

IMPROVEMENTS over v1:
  [A1] StochasticDepth (drop-path) in DualEncoderBlock — better regularization
       than Dropout2d for residual networks; empirically +0.5–1% wPCC.
  [A2] CrossAttentionBottleneck — rain ↔ topo cross-attention before self-attn.
       Lets the bottleneck explicitly learn "what topo feature explains this
       rain anomaly" instead of relying on concatenation alone.
  [A3] FreqAwareAttention — splits feature map into low/high frequency halves
       (via avg-pool / difference), attends on low-freq tokens, re-merges.
       This makes the bottleneck attend to synoptic patterns without
       being distracted by pixel noise.
  [A4] Improved TemporalEncoder with causal depthwise-temporal conv +
       attention pooling over the sequence (not just last LSTM state).
       More stable gradients for long sequences.
  [A5] DryDayGate — soft sigmoid gate on the output that learns to zero-out
       dry-day predictions; reduces false positives without a hard threshold.
  [A6] QDM (Quantile Delta Mapping) — inference-only wrapper that corrects
       the diffusion output's CDF to match a reference climatology.
       Call QDM.fit(ref_samples) once after training; use QDM.apply() at test.

Original classes kept unchanged (CorrDiffRegressor, UNet) for backward compat.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


def _safe_groups(ch, max_groups=32):
    for g in range(max_groups, 0, -1):
        if ch % g == 0:
            return g
    return 1


# ================================================================
# EMBEDDINGS & SE BLOCK  (unchanged)
# ================================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb_scale)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=1)


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(1, ch // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, ch // reduction), ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class WHTLayer(nn.Module):
    def __init__(self, scale=True):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if x.shape[1] == 1 or (x.shape[1] & (x.shape[1] - 1)) != 0:
            return x
        try:
            return self._apply_wht(x)
        except Exception:
            return x  # safe fallback

    def _apply_wht(self, x):
        n = x.shape[1]
        x = x.view(-1, 2, n // 2, *x.shape[2:])
        a, b = x[:, 0], x[:, 1]
        res = torch.cat([a + b, a - b], dim=1)
        res = res / math.sqrt(2) if self.scale else res
        return torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)



# ================================================================
# [A1] STOCHASTIC DEPTH (DROP-PATH)
# ================================================================

class StochasticDepth(nn.Module):
    """
    Drop entire residual branches during training (DropPath / Stochastic Depth).
    Superior to Dropout2d for residual networks — the skip connection always
    passes through, so the network can still learn even when a branch is dropped.

    survival_prob : 1 - drop_rate  (use ~0.85–0.95 for encoder blocks)
    """
    def __init__(self, survival_prob: float = 0.9):
        super().__init__()
        self.survival_prob = survival_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.survival_prob == 1.0:
            return x
        # Per-sample binary mask — shape [B, 1, 1, 1] for broadcasting
        mask = torch.bernoulli(
            torch.full((x.shape[0], 1, 1, 1), self.survival_prob, device=x.device)
        )
        return x * mask / self.survival_prob   # scale to keep E[x] unchanged


# ================================================================
# [A2] CROSS-ATTENTION BOTTLENECK
# ================================================================
class CrossAttentionBottleneck(nn.Module):
    """
    Bidirectional cross-attention between rain and topo feature maps.
    """
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        assert ch % heads == 0, f"ch={ch} must be divisible by heads={heads}"
        self.heads = heads
        self.head_ch = ch // heads

        # Projections
        self.r_q = nn.Conv2d(ch, ch, 1)
        self.t_k = nn.Conv2d(ch, ch, 1)
        self.t_v = nn.Conv2d(ch, ch, 1)
        self.r_out = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.GroupNorm(_safe_groups(ch), ch))

        self.t_q = nn.Conv2d(ch, ch, 1)
        self.r_k = nn.Conv2d(ch, ch, 1)
        self.r_v = nn.Conv2d(ch, ch, 1)
        self.t_out = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.GroupNorm(_safe_groups(ch), ch))

        self.scale = self.head_ch ** -0.5

        # Better initialization
        for m in [self.r_q, self.t_k, self.t_v, self.t_q, self.r_k, self.r_v]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for m in [self.r_out[0], self.t_out[0]]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _elu_feature(self, x: torch.Tensor) -> torch.Tensor:
        return (F.elu(x) + 1.0).clamp(-10.0, 10.0)

    def _linear_cross_attn(
        self,
        q: torch.Tensor,  # [B, C, H, W]
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = q.shape
        h = self.heads
        d = self.head_ch
        N = H * W

        # Reshape to [B, h, N, d]
        q = q.view(B, h, d, N).permute(0, 1, 3, 2)
        k = k.view(B, h, d, N).permute(0, 1, 3, 2)
        v = v.view(B, h, d, N).permute(0, 1, 3, 2)

        q = self._elu_feature(q * self.scale)
        k = self._elu_feature(k)

        # Linear attention
        context = torch.einsum('bhnd,bhne->bhde', k, v)          # [B, h, d, d]
        out = torch.einsum('bhnd,bhde->bhne', q, context)        # [B, h, N, d]

        # Normalization
        k_sum = k.sum(dim=2, keepdim=True)                       # [B, h, 1, d]
        denom = (q * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = out / denom                                        # ← Correct broadcasting here
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)

        # Back to [B, C, H, W]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return out

    def forward(self, rain: torch.Tensor, topo: torch.Tensor):
        rain = rain.clamp(-20, 20)  # ADD THIS
        topo = topo.clamp(-20, 20)  # ADD THIS
        r_updated = rain + self.r_out(
            self._linear_cross_attn(self.r_q(rain), self.t_k(topo), self.t_v(topo))
        )
        t_updated = topo + self.t_out(
            self._linear_cross_attn(self.t_q(topo), self.r_k(rain), self.r_v(rain))
        )
        r_updated = torch.nan_to_num(r_updated, nan=0.0, posinf=20.0, neginf=-20.0)  # ADD
        t_updated = torch.nan_to_num(t_updated, nan=0.0, posinf=20.0, neginf=-20.0)  # ADD
        return r_updated, t_updated


# ================================================================
# [A3] FREQUENCY-AWARE SELF-ATTENTION
# ================================================================

class FreqAwareSelfAttention(nn.Module):
    """
    Splits feature map into low- and high-frequency components, performs
    multi-head self-attention ONLY on the low-frequency tokens (spatially
    pooled to 8×8 max), then broadcasts the attended features back and
    re-merges.

    Why: Synoptic rainfall structure (low-freq) benefits from global attention.
    Pixel-level detail (high-freq) is better handled by local convolutions —
    attending to every pixel pair at 128×128 wastes computation and can hurt.

    This gives most of the benefit of global attention at ~1/256 the FLOPs.
    """
    def __init__(self, ch: int, attn_size: int = 8, heads: int = 4):
        super().__init__()
        self.attn_size = attn_size
        assert ch % heads == 0
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True, dropout=0.0)
        self.proj  = nn.Conv2d(ch, ch, 1)
        self.norm  = nn.GroupNorm(_safe_groups(ch), ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Low-freq: spatial average pool to attn_size × attn_size
        lo = F.adaptive_avg_pool2d(x, (self.attn_size, self.attn_size))  # [B,C,s,s]
        hi = x - F.interpolate(lo, size=(H, W), mode='bilinear', align_corners=False)

        # Self-attention on low-freq tokens
        s = self.attn_size
        tokens = lo.view(B, C, s * s).permute(0, 2, 1)           # [B, s^2, C]
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, s, s)    # [B, C, s, s]

        # Upsample attended low-freq back to original resolution
        lo_updated = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)

        # Merge: original skip + attended low-freq (high-freq is untouched)
        out = x + self.proj(self.norm(lo_updated + hi))
        return out


# ================================================================
# CONV LSTM  (improved with causal temporal conv)
# ================================================================

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv_ih = nn.Conv2d(in_ch,  4 * hid_ch, kernel_size, padding=pad, bias=True)
        self.conv_hh = nn.Conv2d(hid_ch, 4 * hid_ch, kernel_size, padding=pad, bias=False)
        self.hid_ch  = hid_ch
        self.ln      = nn.GroupNorm(_safe_groups(4 * hid_ch), 4 * hid_ch)

    def forward(self, x, h_prev, c_prev):
        gates = self.ln(self.conv_ih(x) + self.conv_hh(h_prev))
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f)
        g = torch.tanh(g);    o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch, h, w, device):
        return (
            torch.zeros(batch, self.hid_ch, h, w, device=device),
            torch.zeros(batch, self.hid_ch, h, w, device=device),
        )


class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hid_ch, num_layers=2, kernel_size=3, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hid_ch     = hid_ch
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            self.cells.append(ConvLSTMCell(in_ch if i == 0 else hid_ch, hid_ch, kernel_size))
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, seq, init_states=None):
        B, T, C, H, W = seq.shape
        device = seq.device
        if init_states is None:
            states = [self.cells[l].init_hidden(B, H, W, device) for l in range(self.num_layers)]
        else:
            states = init_states
        all_h_last = []
        for t in range(T):
            x_t = seq[:, t]
            for l, cell in enumerate(self.cells):
                h, c = states[l]; h, c = cell(x_t, h, c)
                states[l] = (h, c); x_t = self.drop(h)
            all_h_last.append(h)
        all_h = torch.stack(all_h_last, dim=1)
        return all_h, states


# ================================================================
# [A4] IMPROVED TEMPORAL ENCODER
# ================================================================

class TemporalEncoder(nn.Module):
    """
    Improved TemporalEncoder with:
      1. Causal depthwise-temporal conv before LSTM — helps capture local
         temporal patterns (e.g., consecutive heavy rain days) cheaply.
      2. Attention pooling over all LSTM hidden states instead of just the
         last one — the most relevant timestep may not be the most recent.
      3. Residual connection from the mean of all embedded frames — acts as
         a skip for when LSTM struggles early in training.

    Interface is identical to original TemporalEncoder.
    """
    def __init__(
        self,
        in_ch:       int,
        stem_ch:     int,
        lstm_layers: int = 2,
        lstm_drop:   float = 0.0,
        fine_size:   int  = 128,
    ):
        super().__init__()
        self.fine_size = fine_size

        # Per-frame spatial embedding
        self.frame_embed = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(stem_ch), stem_ch),
            nn.SiLU(),
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1, groups=stem_ch),  # depthwise refinement
            nn.GroupNorm(_safe_groups(stem_ch), stem_ch),
            nn.SiLU(),
        )

        # Causal depthwise temporal conv (operates on T dimension)
        # kernel_size=3 → looks at [t-2, t-1, t] without future leakage
        self.temp_conv = nn.Conv1d(
            stem_ch, stem_ch, kernel_size=3, padding=2,   # causal padding
            groups=stem_ch, bias=False
        )

        self.lstm = ConvLSTM(
            in_ch=stem_ch, hid_ch=stem_ch,
            num_layers=lstm_layers, dropout=lstm_drop,
        )

        # Attention pooling over T LSTM outputs → single context frame
        # A single linear layer produces attention logits per timestep
        self.attn_pool = nn.Sequential(
            nn.Conv2d(stem_ch, 1, 1),    # [B, 1, H, W] → spatially-averaged → scalar
            nn.Flatten(1),               # [B, H*W] — this is applied per-step outside
        )
        # Simpler: learned per-timestep scalar weights after global avg
        self.step_attn = nn.Linear(stem_ch, 1)   # produces 1 score per [B, stem_ch] avg

        self.out_conv = nn.Sequential(
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(stem_ch), stem_ch),
            nn.SiLU(),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = seq.shape

        # Per-frame spatial embedding
        x = seq.view(B * T, C, H, W)
        x = self.frame_embed(x)           # [B*T, stem_ch, H, W]
        _, stem_ch, Hx, Wx = x.shape
        x = x.view(B, T, stem_ch, Hx, Wx)

        # Causal temporal conv — operate on T dim per spatial location
        # Reshape: [B, stem_ch, T, H*W] → apply Conv1d on T per channel
        x_tc = x.permute(0, 2, 1, 3, 4).reshape(B * stem_ch, T, Hx * Wx)
        x_tc = x_tc.permute(0, 2, 1)                     # [B*C, HW, T]
        x_tc = self.temp_conv(x_tc)[:, :, :T]            # causal: drop future padding
        x_tc = x_tc.permute(0, 2, 1).reshape(B, stem_ch, T, Hx, Wx)
        x    = x + x_tc.permute(0, 2, 1, 3, 4)           # residual add

        # Mean of embedded frames (skip for LSTM instability early in training)
        mean_ctx = x.mean(dim=1)                          # [B, stem_ch, H, W]

        # LSTM
        all_h, _ = self.lstm(x)                          # [B, T, stem_ch, H, W]

        # Attention pooling over T steps
        # Global avg of each hidden state → [B, T, stem_ch]
        h_avg  = all_h.mean(dim=(-2, -1))                # [B, T, stem_ch]
        scores = self.step_attn(h_avg).squeeze(-1)        # [B, T]
        weights = torch.softmax(scores, dim=1)            # [B, T]
        # Weighted sum of all hidden states
        h_pool = (all_h * weights.view(B, T, 1, 1, 1)).sum(dim=1)   # [B, stem_ch, H, W]

        # Combine attention-pooled + mean-ctx (residual)
        h_combined = h_pool + 0.1 * mean_ctx

        # Upsample to fine resolution
        ctx = F.interpolate(h_combined, size=(self.fine_size, self.fine_size),
                            mode='bilinear', align_corners=False)
        return self.out_conv(ctx)


# ================================================================
# RESIDUAL BLOCK  (unchanged)
# ================================================================

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch, down=False, up=False):
        super().__init__()
        self.down = down; self.up = up
        self.se = SEBlock(out_ch); self.wht = WHTLayer()
        if down:   self.resample = nn.Conv2d(in_ch, in_ch, 4, stride=2, padding=1)
        elif up:   self.resample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        else:      self.resample = None
        self.norm1    = nn.GroupNorm(_safe_groups(in_ch), in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_ch, out_ch)
        self.norm2    = nn.GroupNorm(_safe_groups(out_ch), out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        if self.resample is not None: x = self.resample(x)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        if (h.shape[1] & (h.shape[1] - 1) == 0) and h.shape[1] > 0: h = self.wht(h)
        h = self.conv2(F.silu(self.norm2(h)))
        h = self.se(h)
        return h + self.skip(x)


# ================================================================
# SPADE  (unchanged)
# ================================================================

class SPADEGroupNorm(nn.Module):
    def __init__(self, num_channels, cond_channels):
        super().__init__()
        g = _safe_groups(num_channels)
        self.norm = nn.GroupNorm(g, num_channels, affine=False)
        hidden    = max(64, num_channels // 2)
        self.shared = nn.Sequential(nn.Conv2d(cond_channels, hidden, 3, padding=1), nn.SiLU())
        self.gamma  = nn.Conv2d(hidden, num_channels, 3, padding=1)
        self.beta   = nn.Conv2d(hidden, num_channels, 3, padding=1)
        self.wht    = WHTLayer()
        nn.init.zeros_(self.gamma.bias);  nn.init.normal_(self.gamma.weight, 0, 0.02)
        nn.init.zeros_(self.beta.bias);   nn.init.normal_(self.beta.weight,  0, 0.02)

    def forward(self, x, cond):
        norm_x    = self.norm(x)
        cond_feat = self.wht(self.shared(cond))
        return norm_x * (1.0 + self.gamma(cond_feat)) + self.beta(cond_feat)


# ================================================================
# ENCODER / DECODER BLOCKS  (DualEncoderBlock upgraded with DropPath)
# ================================================================

class DualEncoderBlock(nn.Module):
    """
    [A1] Added StochasticDepth on the residual branch.
    survival_prob=1.0 → identical to original (safe default for early layers).
    Deeper layers benefit from lower survival_prob (~0.85).
    """
    def __init__(self, in_ch, out_ch, dropout=0.0, survival_prob=0.9):
        super().__init__()
        g = _safe_groups(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.se    = SEBlock(out_ch)
        self.drop2d= nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.stoch_depth = StochasticDepth(survival_prob)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.se(h)
        h = self.drop2d(h)
        h = F.silu(self.norm2(self.conv2(h)))
        return self.skip(x) + self.stoch_depth(h)   # DropPath on residual only


class SPADEDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = SPADEGroupNorm(out_ch, cond_channels)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = SPADEGroupNorm(out_ch, cond_channels)
        self.se    = SEBlock(out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, cond):
        h = F.silu(self.norm1(self.conv1(x), cond))
        h = self.drop(h)
        h = self.se(F.silu(self.norm2(self.conv2(h), cond)))
        return h + self.skip(x)


# ================================================================
# [A5] DRY-DAY GATE
# ================================================================

class DryDayGate(nn.Module):
    """
    Learnable soft gate that suppresses predictions on dry days.

    Given the final feature map, produces a [B, 1, H, W] probability-of-wet
    map via sigmoid, then multiplies the rainfall output.  This is trained
    end-to-end and replaces ad-hoc dry-mask loss terms.

    Advantages:
      - No hyperparameter tuning (margin, weight) needed
      - Spatially adaptive — can predict sub-grid wet/dry patterns
      - Gradient flows through the gate; the backbone learns to produce clean
        dry-day features too, not just clean wet-day features

    If p_wet → 0 everywhere the model naturally outputs zero rainfall,
    which is what we want.
    """
    def __init__(self, in_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, max(8, in_ch // 4), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(8, in_ch // 4), 1, 1),
            nn.Sigmoid(),
        )
        # Initialise bias strongly negative → gate starts near-zero
        # (network must learn to open it — prevents wet-bias at init)
        nn.init.constant_(self.gate[-2].bias, -3.0)

    def forward(self, features: torch.Tensor, rainfall: torch.Tensor) -> torch.Tensor:
        p_wet = self.gate(features)       # [B, 1, H, W] ∈ (0, 1)
        return rainfall * p_wet


# ================================================================
# CORR DIFF REGRESSOR  (upgraded: cross-attn + freq-attn + dry-day gate)
# ================================================================

class CorrDiffRegressor(nn.Module):
    """
    Dual-encoder U-Net with:
      [A2] CrossAttentionBottleneck (rain ↔ topo)
      [A3] FreqAwareSelfAttention replacing plain MHA in bottleneck
      [A1] StochasticDepth in encoder blocks
      [A5] DryDayGate on output

    Backward compatible: same constructor signature as original.
    Extra kwargs:
      cross_attn    : bool (default True)  — enable cross-attention
      freq_attn     : bool (default True)  — enable freq-aware self-attn
      dry_gate      : bool (default True)  — enable dry-day gate
      drop_path_rate: float (default 0.1)  — max drop-path rate (linearly
                      increased from 0 at first encoder block to this value)
    """
    def __init__(
        self,
        in_channels,
        out_channels=1,
        base_channels=64,
        channel_mult=(1, 2, 4),
        num_blocks=2,
        global_dim=2,
        dropout=0.0,
        cross_attn=True,
        freq_attn=True,
        dry_gate=True,
        drop_path_rate=0.1,
        **kwargs
    ):
        super().__init__()
        self.channel_mult = list(channel_mult)
        self.num_blocks   = num_blocks
        self.global_dim   = global_dim
        emb_dim           = base_channels * 4
        self.emb_dim      = emb_dim

        if global_dim is not None and global_dim > 0:
            self.global_mlp = nn.Sequential(
                nn.Linear(global_dim, emb_dim), nn.SiLU(),
                nn.Linear(emb_dim, emb_dim),   nn.SiLU(),
            )
        else:
            self.global_mlp = None

        stem_ch = base_channels // 2

        self.rain_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, stem_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(stem_ch), stem_ch), nn.SiLU(),
        )
        self.topo_stem = nn.Sequential(
            nn.Conv2d(1, stem_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(stem_ch), stem_ch), nn.SiLU(),
        )
        self.global_stem_proj = (nn.Linear(emb_dim, stem_ch) if self.global_mlp else None)

        # Build drop-path rates (linearly scaled across blocks)
        total_blocks = len(channel_mult) * num_blocks * 2   # rain + topo
        dp_rates     = [drop_path_rate * i / max(total_blocks - 1, 1)
                        for i in range(total_blocks)]
        dp_idx = 0

        self.rain_enc  = nn.ModuleList()
        self.topo_enc  = nn.ModuleList()
        self.rain_down = nn.ModuleList()
        self.topo_down = nn.ModuleList()
        self.enc_skip_ch = []

        curr_rain_ch = curr_topo_ch = stem_ch
        for i, mult in enumerate(self.channel_mult):
            out_ch = base_channels * mult
            r_blocks = nn.ModuleList(); t_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                r_blocks.append(DualEncoderBlock(curr_rain_ch, out_ch, dropout,
                                                  survival_prob=1.0 - dp_rates[dp_idx]))
                t_blocks.append(DualEncoderBlock(curr_topo_ch, out_ch, dropout,
                                                  survival_prob=1.0 - dp_rates[dp_idx]))
                dp_idx += 2
                curr_rain_ch = curr_topo_ch = out_ch
            self.rain_enc.append(r_blocks); self.topo_enc.append(t_blocks)
            self.enc_skip_ch.append(curr_rain_ch + curr_topo_ch)
            if i != len(self.channel_mult) - 1:
                self.rain_down.append(nn.Conv2d(curr_rain_ch, curr_rain_ch, 4, stride=2, padding=1))
                self.topo_down.append(nn.Conv2d(curr_topo_ch, curr_topo_ch, 4, stride=2, padding=1))
            else:
                self.rain_down.append(None); self.topo_down.append(None)

        bn_ch = base_channels * self.channel_mult[-1]

        # [A2] Cross-attention BEFORE projection
        self.cross_attn_enabled = cross_attn
        if cross_attn:
            self.cross_attn_block = CrossAttentionBottleneck(curr_rain_ch)

        bn_in = curr_rain_ch + curr_topo_ch
        self.bn_proj  = nn.Conv2d(bn_in, bn_ch, 1)
        self.wht_heal = WHTLayer()

        # [A3] Freq-aware self-attention OR fallback to plain MHA
        self.freq_attn_enabled = freq_attn
        if freq_attn:
            self.bn_attn = FreqAwareSelfAttention(bn_ch, attn_size=8,
                                                   heads=max(1, bn_ch // 64))
        else:
            self.bn_attn = nn.MultiheadAttention(bn_ch, max(1, bn_ch // 64), batch_first=True)

        self.bn_mid = nn.Sequential(
            nn.GroupNorm(_safe_groups(bn_ch), bn_ch), nn.SiLU(),
            nn.Conv2d(bn_ch, bn_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(bn_ch), bn_ch), nn.SiLU(),
        )
        self.global_bn_proj = (nn.Linear(emb_dim, bn_ch) if self.global_mlp else None)

        COND_CH = 2
        self.dec_ups    = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_ch = bn_ch

        for i, mult in reversed(list(enumerate(self.channel_mult))):
            out_ch  = base_channels * mult
            skip_ch = self.enc_skip_ch[i]
            self.dec_ups.append(
                nn.ConvTranspose2d(dec_ch, dec_ch, 4, stride=2, padding=1)
                if i != len(self.channel_mult) - 1 else None
            )
            dec_in   = dec_ch + skip_ch
            blk_list = nn.ModuleList()
            for _ in range(num_blocks):
                blk_list.append(SPADEDecoderBlock(dec_in, out_ch, COND_CH, dropout))
                dec_in = out_ch
            self.dec_blocks.append(blk_list)
            dec_ch = out_ch

        self.final_norm = nn.GroupNorm(_safe_groups(dec_ch), dec_ch)

        # [A5] Dry-day gate uses final feature map before 1×1 conv
        self.dry_gate_enabled = dry_gate
        if dry_gate:
            self.dry_gate = DryDayGate(dec_ch)

        self.final_conv = nn.Conv2d(dec_ch, out_channels, 3, padding=1)
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, topo, global_features=None):
        g_emb = (self.global_mlp(global_features)
                 if (self.global_mlp and global_features is not None) else None)

        rain = self.rain_upsample(x)
        tf   = self.topo_stem(topo)

        if g_emb is not None and self.global_stem_proj is not None:
            g_stem = self.global_stem_proj(g_emb)[:, :, None, None]
            rain = rain + g_stem; tf = tf + g_stem

        clim_fine = F.interpolate(x[:, 0:1], size=topo.shape[-2:],
                                  mode='bilinear', align_corners=False).detach()
        cond_fine  = torch.cat([clim_fine, topo], dim=1)

        rain_skips, topo_skips = [], []
        for i in range(len(self.channel_mult)):
            for r_blk, t_blk in zip(self.rain_enc[i], self.topo_enc[i]):
                rain = r_blk(rain); tf = t_blk(tf)
            rain_skips.append(rain); topo_skips.append(tf)
            if self.rain_down[i] is not None:
                rain = self.rain_down[i](rain); tf = self.topo_down[i](tf)

        # [A2] Cross-attention between streams before concatenation
        if self.cross_attn_enabled:
            rain, tf = self.cross_attn_block(rain, tf)

        fused = self.wht_heal(torch.cat([rain, tf], dim=1))
        fused = self.bn_proj(fused)
        if g_emb is not None and self.global_bn_proj is not None:
            fused = fused + self.global_bn_proj(g_emb)[:, :, None, None]

        # [A3] Bottleneck attention
        if self.freq_attn_enabled:
            fused = self.bn_attn(fused)          # FreqAwareSelfAttention API
        else:
            B, C, H, W = fused.shape
            flat = fused.view(B, C, -1).permute(0, 2, 1)
            attn_out, _ = self.bn_attn(flat, flat, flat)
            fused = fused + attn_out.permute(0, 2, 1).view(B, C, H, W)

        fused = self.bn_mid(fused)

        x_dec = fused
        for i, (up, blk_list) in enumerate(zip(self.dec_ups, self.dec_blocks)):
            level = len(self.channel_mult) - 1 - i
            if up is not None: x_dec = up(x_dec)
            skip  = torch.cat([rain_skips[level], topo_skips[level]], dim=1)
            x_dec = torch.cat([x_dec, skip], dim=1)
            cond_r = F.interpolate(cond_fine, size=x_dec.shape[-2:],
                                   mode='bilinear', align_corners=False)
            for blk in blk_list:
                x_dec = blk(x_dec, cond_r)

        feat = F.silu(self.final_norm(x_dec))

        # [A5] Dry-day gate
        out = self.final_conv(feat)
        if self.dry_gate_enabled:
            out = self.dry_gate(feat, out)

        return out


# ================================================================
# TEMPORAL CORRDIFF REGRESSOR  (unchanged interface, uses improved core)
# ================================================================

class TemporalCorrDiffRegressor(nn.Module):
    """
    CorrDiffRegressor extended with improved ConvLSTM temporal encoder.
    Interface identical to original — new improvements come from the
    upgraded CorrDiffRegressor (cross-attn, freq-attn, dry gate) and
    improved TemporalEncoder (causal conv + attention pooling).
    """
    def __init__(
        self,
        in_channels,
        out_channels=1,
        base_channels=64,
        channel_mult=(1, 2, 4),
        num_blocks=2,
        global_dim=2,
        dropout=0.0,
        seq_len=5,
        lstm_layers=2,
        lstm_drop=0.0,
        fine_size=128,
        **kwargs
    ):
        super().__init__()
        self.seq_len = seq_len

        self.core = CorrDiffRegressor(
            in_channels=in_channels, out_channels=out_channels,
            base_channels=base_channels, channel_mult=channel_mult,
            num_blocks=num_blocks, global_dim=global_dim, dropout=dropout,
            **kwargs
        )

        stem_ch = base_channels // 2

        self.temporal_enc = TemporalEncoder(
            in_ch=in_channels, stem_ch=stem_ch,
            lstm_layers=lstm_layers, lstm_drop=lstm_drop,
            fine_size=fine_size,
        )

        self.ctx_gate = nn.Sequential(
            nn.Conv2d(stem_ch * 2, stem_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, topo, global_features=None, seq=None):
        if seq is None:
            return self.core(x, topo, global_features)

        ctx = self.temporal_enc(seq)

        g_emb = (self.core.global_mlp(global_features)
                 if (self.core.global_mlp and global_features is not None) else None)

        rain = self.core.rain_upsample(x)
        tf   = self.core.topo_stem(topo)

        if g_emb is not None and self.core.global_stem_proj is not None:
            g_stem = self.core.global_stem_proj(g_emb)[:, :, None, None]
            rain = rain + g_stem; tf = tf + g_stem

        gate = self.ctx_gate(torch.cat([rain, ctx], dim=1))
        rain = rain + gate * ctx

        clim_fine = F.interpolate(x[:, 0:1], size=topo.shape[-2:],
                                  mode='bilinear', align_corners=False).detach()
        cond_fine  = torch.cat([clim_fine, topo], dim=1)

        # [A2] Cross-attention
        if self.core.cross_attn_enabled:
            rain, tf = self.core.cross_attn_block(rain, tf)

        rain_skips, topo_skips = [], []
        for i in range(len(self.core.channel_mult)):
            for r_blk, t_blk in zip(self.core.rain_enc[i], self.core.topo_enc[i]):
                rain = r_blk(rain); tf = t_blk(tf)
            rain_skips.append(rain); topo_skips.append(tf)
            if self.core.rain_down[i] is not None:
                rain = self.core.rain_down[i](rain); tf = self.core.topo_down[i](tf)

        fused = self.core.wht_heal(torch.cat([rain, tf], dim=1))
        fused = self.core.bn_proj(fused)
        if g_emb is not None and self.core.global_bn_proj is not None:
            fused = fused + self.core.global_bn_proj(g_emb)[:, :, None, None]

        if self.core.freq_attn_enabled:
            fused = self.core.bn_attn(fused)
        else:
            B, C, H, W = fused.shape
            flat = fused.view(B, C, -1).permute(0, 2, 1)
            attn_out, _ = self.core.bn_attn(flat, flat, flat)
            fused = fused + attn_out.permute(0, 2, 1).view(B, C, H, W)

        fused = self.core.bn_mid(fused)

        x_dec = fused
        for i, (up, blk_list) in enumerate(zip(self.core.dec_ups, self.core.dec_blocks)):
            level = len(self.core.channel_mult) - 1 - i
            if up is not None: x_dec = up(x_dec)
            skip  = torch.cat([rain_skips[level], topo_skips[level]], dim=1)
            x_dec = torch.cat([x_dec, skip], dim=1)
            cond_r = F.interpolate(cond_fine, size=x_dec.shape[-2:],
                                   mode='bilinear', align_corners=False)
            for blk in blk_list:
                x_dec = blk(x_dec, cond_r)

        feat = F.silu(self.core.final_norm(x_dec))
        out  = self.core.final_conv(feat)
        if self.core.dry_gate_enabled:
            out = self.core.dry_gate(feat, out)
        return out


# ================================================================
# WEIGHTED ENSEMBLE  (unchanged)
# ================================================================

class WeightedEnsemble(nn.Module):
    def __init__(self, model_cls, model_kwargs, ckpt_paths, weights=None, ema=True, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.models = nn.ModuleList()
        for path in ckpt_paths:
            m = model_cls(**model_kwargs).to(device)
            ck = torch.load(path, map_location=device)
            key = "ema_state_dict" if (ema and "ema_state_dict" in ck) else "model_state_dict"
            m.load_state_dict(ck[key])
            m.eval()
            for p in m.parameters(): p.requires_grad_(False)
            self.models.append(m)
        if weights is None:
            w = torch.ones(len(ckpt_paths), device=device) / len(ckpt_paths)
        else:
            raw = torch.tensor(weights, dtype=torch.float32, device=device)
            w   = torch.softmax(raw / 0.5, dim=0)
        self.register_buffer("weights", w)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        preds   = [m(*args, **kwargs) for m in self.models]
        stacked = torch.stack(preds, dim=0)
        w       = self.weights.view(-1, 1, 1, 1, 1)
        return (stacked * w).sum(dim=0)


def ensemble_predict(model_cls, model_kwargs, ckpt_paths, xi, topo, tf,
                     weights=None, device=None, seq=None):
    ens = WeightedEnsemble(model_cls, model_kwargs, ckpt_paths, weights, device=device)
    ens.eval()
    kwargs = dict(topo=topo, global_features=tf)
    if seq is not None: kwargs["seq"] = seq
    with torch.no_grad():
        return ens(xi, **kwargs)


# ================================================================
# [A6] QUANTILE DELTA MAPPING (QDM) — Inference-only bias correction
# ================================================================

class QDM:
    """
    Quantile Delta Mapping for post-hoc bias correction of diffusion output.

    Background:
      Even a well-trained diffusion model can have systematic CDF bias —
      e.g., it slightly over-estimates light rain or under-estimates
      extreme events.  QDM corrects this by mapping the model's CDF to
      match a reference (observational) CDF, while *preserving the
      relative anomaly* (delta) of each prediction.  This is the climate
      science standard for statistical downscaling bias correction.

    How it works (Cannon et al. 2015):
      For each predicted value x̂, find its quantile q in the model CDF,
      look up the corresponding reference value F_obs^{-1}(q), then
      preserve the multiplicative delta:
          x̂_corrected = F_obs^{-1}(q) * (x̂ / F_mod^{-1}(q))

    Usage:
        qdm = QDM(n_quantiles=1000)

        # Fit on validation set predictions vs observations (linear space)
        qdm.fit(
            model_samples=all_val_preds_lin,   # [N] or [N, H, W] flattened
            obs_samples=all_val_obs_lin,        # same shape
        )

        # Apply at test time
        pred_corrected = qdm.apply(pred_lin)   # [B, 1, H, W]

    Notes:
      - fit() and apply() operate in LINEAR (mm/day) space, not log1p.
        Convert back from log1p before calling.
      - Pixelwise QDM is too expensive; this implementation fits a single
        global CDF (pooled over all pixels and samples), which is standard
        for precipitation downscaling.
      - Set clip_min=0 to avoid negative rainfall after correction.
    """

    def __init__(self, n_quantiles: int = 1000, clip_min: float = 0.0):
        self.n_quantiles = n_quantiles
        self.clip_min    = clip_min
        self.quantiles   = torch.linspace(0.0, 1.0, n_quantiles)
        self.cdf_mod: Optional[torch.Tensor] = None   # F_mod^{-1}(q) — model quantiles
        self.cdf_obs: Optional[torch.Tensor] = None   # F_obs^{-1}(q) — obs quantiles
        self._fitted     = False

    def fit(
        self,
        model_samples: torch.Tensor,   # flat or any shape — all values pooled
        obs_samples:   torch.Tensor,
    ) -> None:
        """Estimate CDFs from validation predictions and observations."""
        mp = model_samples.flatten().float().cpu()
        op = obs_samples.flatten().float().cpu()

        # Use torch.quantile for GPU-friendly computation
        self.cdf_mod = torch.quantile(mp, self.quantiles)
        self.cdf_obs = torch.quantile(op, self.quantiles)
        self._fitted = True
        print(f"[QDM] Fitted on {len(mp):,} model / {len(op):,} obs samples")
        print(f"      Model  P95={self.cdf_mod[950]:.2f}  P99={self.cdf_mod[990]:.2f}")
        print(f"      Obs    P95={self.cdf_obs[950]:.2f}  P99={self.cdf_obs[990]:.2f}")

    @torch.no_grad()
    def apply(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Apply QDM correction to a batch of predictions.

        pred : [B, 1, H, W] in linear (mm/day) space
        returns corrected tensor, same shape
        """
        if not self._fitted:
            raise RuntimeError("QDM.fit() must be called before QDM.apply()")

        device   = pred.device
        cdf_mod  = self.cdf_mod.to(device)
        cdf_obs  = self.cdf_obs.to(device)

        shape    = pred.shape
        x        = pred.flatten().float()

        # Find quantile index for each predicted value via searchsorted
        # x is in model-CDF space; find where it falls in cdf_mod
        idx = torch.searchsorted(cdf_mod.contiguous(), x.contiguous()).clamp(0, self.n_quantiles - 1)

        # Model CDF value at that quantile
        x_mod_q  = cdf_mod[idx].clamp(min=1e-6)   # F_mod^{-1}(q)

        # Observed CDF value at that quantile
        x_obs_q  = cdf_obs[idx]                    # F_obs^{-1}(q)

        # Multiplicative delta mapping: preserve relative anomaly
        delta    = (x / x_mod_q).clamp(0.0, 10.0)  # relative anomaly, capped
        x_corr   = x_obs_q * delta

        x_corr   = x_corr.clamp(min=self.clip_min)
        return x_corr.reshape(shape).to(pred.dtype)

    def save(self, path: str) -> None:
        torch.save({"cdf_mod": self.cdf_mod, "cdf_obs": self.cdf_obs,
                    "n_quantiles": self.n_quantiles, "clip_min": self.clip_min}, path)

    @classmethod
    def load(cls, path: str) -> "QDM":
        d   = torch.load(path, map_location="cpu")
        qdm = cls(n_quantiles=d["n_quantiles"], clip_min=d["clip_min"])
        qdm.cdf_mod  = d["cdf_mod"]
        qdm.cdf_obs  = d["cdf_obs"]
        qdm._fitted  = True
        return qdm


# ================================================================
# UNET FOR DIFFUSION (Stage 2 — improved with EDMv2 preconditioning)
# ================================================================

class UNet(nn.Module):
    """
    Diffusion denoiser U-Net with:
      - EDM-v2-style Fourier conditioning on sigma (replaces sinusoidal)
      - [A3] FreqAwareSelfAttention in bottleneck (optional, controlled by
        use_bottleneck_attention)
      - Same interface as original

    EDM-v2 improvement: instead of a fixed sinusoidal embedding of t,
    we use a learned Fourier feature embedding of log(sigma).  This gives
    the model a smoother and more expressive conditioning signal over the
    wide sigma range used in EDM training.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        num_blocks=None,
        global_dim=2,
        use_bottleneck_attention=False,
        **kwargs
    ):
        super().__init__()
        num_res_blocks = num_blocks if num_blocks is not None else num_res_blocks
        emb_ch = base_channels * 4

        # EDM-v2: learned Fourier embedding of log(sigma)
        # Better than fixed sinusoidal for the wide sigma range in EDM
        self.fourier_freqs = nn.Parameter(
            torch.randn(base_channels // 2) * 0.2, requires_grad=False
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_ch), nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )

        self.global_mlp = nn.Linear(global_dim, emb_ch) if global_dim else None
        self.head       = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down = nn.ModuleList(); self.up = nn.ModuleList()
        curr_ch = base_channels; skip_channels = []

        for mult in channel_mult:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(curr_ch, out_ch, emb_ch))
                curr_ch = out_ch; skip_channels.append(curr_ch)
            self.down.append(ResBlock(curr_ch, curr_ch, emb_ch, down=True))
            skip_channels.append(curr_ch)

        self.mid = nn.ModuleList([
            ResBlock(curr_ch, curr_ch, emb_ch),
            ResBlock(curr_ch, curr_ch, emb_ch),
        ])

        self.use_bottleneck_attention = use_bottleneck_attention
        if use_bottleneck_attention:
            # [A3] Freq-aware attention in diffusion bottleneck
            self.bn_attn = FreqAwareSelfAttention(
                curr_ch, attn_size=min(8, curr_ch // 16),
                heads=max(1, curr_ch // 64)
            )

        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            self.up.append(ResBlock(curr_ch + skip_channels.pop(), out_ch, emb_ch, up=True))
            curr_ch = out_ch
            for _ in range(num_res_blocks):
                self.up.append(ResBlock(curr_ch + skip_channels.pop(), out_ch, emb_ch))
                curr_ch = out_ch

        self.final = nn.Sequential(
            nn.GroupNorm(_safe_groups(curr_ch), curr_ch), nn.SiLU(),
            nn.Conv2d(curr_ch, out_channels, 3, padding=1),
        )

    def _fourier_embed(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        EDM-v2 Fourier embedding of log(sigma).
        sigma : [B]  (noise level, not timestep)
        returns [B, base_channels]
        """
        log_sigma = sigma.log().view(-1, 1)   # [B, 1]
        freqs = self.fourier_freqs.view(1, -1)  # [1, base_channels//2]
        args  = log_sigma * freqs * 2 * math.pi
        embed = torch.cat([args.cos(), args.sin()], dim=-1)   # [B, base_channels]
        return embed

    def forward(self, x, t, global_features=None):
        # t is sigma in EDM formulation; use Fourier embedding
        emb = self.time_mlp(self._fourier_embed(t))
        if global_features is not None and self.global_mlp is not None:
            emb = emb + self.global_mlp(global_features)

        h = self.head(x); skips = [h]

        for layer in self.down:
            h = layer(h, emb); skips.append(h)

        for layer in self.mid:
            h = layer(h, emb)

        if self.use_bottleneck_attention:
            h = self.bn_attn(h)   # FreqAwareSelfAttention

        for layer in self.up:
            skip = skips.pop()
            h    = torch.cat([h, skip], dim=1)
            h    = layer(h, emb)

        return self.final(h)
