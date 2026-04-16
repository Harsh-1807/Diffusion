# -*- coding: utf-8 -*-
"""
CorrDiff Network Architecture (Stage 1 & 2) — WITH TEMPORAL + ENSEMBLE
=======================================================================

NEW in this version:
  [T1] ConvLSTMCell  — single spatiotemporal recurrent cell
  [T2] ConvLSTM      — multi-layer ConvLSTM stack
  [T3] TemporalEncoder — wraps ConvLSTM, produces a spatial context tensor
  [T4] TemporalCorrDiffRegressor — full regressor that accepts a sequence of
       T coarse frames and uses TemporalEncoder to fuse them before decoding
  [E1] WeightedEnsemble — loads N checkpoints, builds validation-weighted
       predictions (used at inference); no DDP required
  [E2] ensemble_predict  — convenience function for ensemble inference

Original classes unchanged (CorrDiffRegressor, UNet) so Stage-2 training
script does not need modification.

Key architecture choices:
  - ConvLSTM hidden size matches rain-stem channels so it plugs in with
    zero extra projection layers.
  - TemporalEncoder returns (h_last, c_last) from the final ConvLSTM layer;
    h_last is used as an additional additive bias after the rain stem.
  - WeightedEnsemble supports both EMA and plain model state dicts.
  - All new classes accept the same (xi, topo, global_features) signature
    as the original CorrDiffRegressor, with an optional `seq` argument for
    the temporal sequence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


def _safe_groups(ch, max_groups=32):
    """Largest divisor of ch that is <= max_groups."""
    for g in range(max_groups, 0, -1):
        if ch % g == 0:
            return g
    return 1


# ================================================================
# EMBEDDINGS & SE BLOCK
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
    """Fast Walsh-Hadamard Transform for channel mixing."""
    def __init__(self, scale=True):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self._apply_wht(x)

    def _apply_wht(self, x):
        n = x.shape[1]
        if n == 1 or (n & (n - 1)) != 0:
            return x   # not a power of 2 — skip safely
        x = x.view(-1, 2, n // 2, *x.shape[2:])
        a, b = x[:, 0], x[:, 1]
        res = torch.cat([a + b, a - b], dim=1)
        return res / math.sqrt(2) if self.scale else res


# ================================================================
# CONV LSTM  (NEW)
# ================================================================

class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell operating on spatial feature maps.

    Input : x     [B, in_ch,  H, W]  — current frame features
            h_prev [B, hid_ch, H, W]  — previous hidden state
            c_prev [B, hid_ch, H, W]  — previous cell state
    Output: h_new, c_new — both [B, hid_ch, H, W]

    Four gates are computed with a single fused convolution for efficiency.
    """
    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        # Input-hidden combined convolution (4 gates fused)
        self.conv_ih = nn.Conv2d(in_ch,      4 * hid_ch, kernel_size, padding=pad, bias=True)
        self.conv_hh = nn.Conv2d(hid_ch,     4 * hid_ch, kernel_size, padding=pad, bias=False)
        self.hid_ch  = hid_ch
        # Layer norm on gates for training stability
        self.ln = nn.GroupNorm(_safe_groups(4 * hid_ch), 4 * hid_ch)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = self.ln(self.conv_ih(x) + self.conv_hh(h_prev))
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch: int, h: int, w: int, device: torch.device):
        return (
            torch.zeros(batch, self.hid_ch, h, w, device=device),
            torch.zeros(batch, self.hid_ch, h, w, device=device),
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM.

    Processes a sequence of spatial feature maps [B, T, C, H, W] and
    returns the hidden states for every timestep plus the final state.

    Args:
        in_ch      : input channel count (= stem channels from rain stem)
        hid_ch     : hidden channels in each layer (= same as in_ch by default)
        num_layers : stacked LSTM depth (2 is a good default)
        kernel_size: spatial convolution kernel (3 recommended)
        dropout    : inter-layer dropout (applied to h between layers)
    """
    def __init__(
        self,
        in_ch:      int,
        hid_ch:     int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hid_ch     = hid_ch

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_in = in_ch if i == 0 else hid_ch
            self.cells.append(ConvLSTMCell(cell_in, hid_ch, kernel_size))

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        seq: torch.Tensor,                            # [B, T, C, H, W]
        init_states: Optional[List[Tuple]] = None,   # optional warm-start
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Returns:
            all_h  : [B, T, hid_ch, H, W]  — hidden states for every step
            states : list of (h, c) for every layer (for warm-start or loss)
        """
        B, T, C, H, W = seq.shape
        device = seq.device

        if init_states is None:
            states = [self.cells[l].init_hidden(B, H, W, device)
                      for l in range(self.num_layers)]
        else:
            states = init_states

        all_h_last = []   # collect final-layer h at each timestep

        for t in range(T):
            x_t = seq[:, t]                    # [B, C, H, W]
            for l, cell in enumerate(self.cells):
                h, c        = states[l]
                h, c        = cell(x_t, h, c)
                states[l]   = (h, c)
                x_t         = self.drop(h)     # output of layer l → input of l+1

            all_h_last.append(h)               # h from the last layer

        all_h = torch.stack(all_h_last, dim=1)  # [B, T, hid_ch, H, W]
        return all_h, states


class TemporalEncoder(nn.Module):
    """
    Converts a sequence of raw coarse precip frames into a single spatial
    context tensor that matches the rain-stem output shape.

    Pipeline:
      seq [B, T, in_ch, H_c, W_c]
        → lightweight conv-embed per frame  [B, T, stem_ch, H_c, W_c]
        → ConvLSTM                          [B, T, stem_ch, H_c, W_c]
        → take last hidden h_T              [B, stem_ch, H_c, W_c]
        → bilinear upsample to fine res     [B, stem_ch, H_f, W_f]
        → output context                    [B, stem_ch, H_f, W_f]

    This context is added to the rain-stem output BEFORE the encoder,
    giving the regressor knowledge of recent rainfall history.
    """
    def __init__(
        self,
        in_ch:       int,           # coarse input channels (e.g. 2)
        stem_ch:     int,           # must match CorrDiffRegressor's stem_ch
        lstm_layers: int = 2,
        lstm_drop:   float = 0.0,
        fine_size:   int  = 128,    # target spatial size after upsampling
    ):
        super().__init__()
        self.fine_size = fine_size

        # Per-frame feature embedding (lightweight)
        self.frame_embed = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(stem_ch), stem_ch),
            nn.SiLU(),
        )

        self.lstm = ConvLSTM(
            in_ch=stem_ch,
            hid_ch=stem_ch,
            num_layers=lstm_layers,
            dropout=lstm_drop,
        )

        # Post-LSTM refinement
        self.out_conv = nn.Sequential(
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(stem_ch), stem_ch),
            nn.SiLU(),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq : [B, T, in_ch, H_c, W_c]  — coarse precip history
        returns context [B, stem_ch, fine_size, fine_size]
        """
        B, T, C, H, W = seq.shape

        # Embed each frame independently
        x = seq.view(B * T, C, H, W)
        x = self.frame_embed(x)                    # [B*T, stem_ch, H, W]
        x = x.view(B, T, -1, H, W)                # [B, T, stem_ch, H, W]

        # Run ConvLSTM — we only keep the final hidden state
        _, states = self.lstm(x)
        h_last = states[-1][0]                     # [B, stem_ch, H, W] (last layer h)

        # Upsample to fine resolution
        ctx = F.interpolate(
            h_last,
            size=(self.fine_size, self.fine_size),
            mode='bilinear',
            align_corners=False,
        )
        return self.out_conv(ctx)                  # [B, stem_ch, fine_size, fine_size]


# ================================================================
# RESIDUAL BLOCK
# ================================================================

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_ch, down=False, up=False):
        super().__init__()
        self.down = down
        self.up   = up
        self.se   = SEBlock(out_ch)
        self.wht  = WHTLayer()

        if down:
            self.resample = nn.Conv2d(in_ch, in_ch, 4, stride=2, padding=1)
        elif up:
            self.resample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        else:
            self.resample = None

        self.norm1    = nn.GroupNorm(_safe_groups(in_ch), in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_ch, out_ch)
        self.norm2    = nn.GroupNorm(_safe_groups(out_ch), out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        if self.resample is not None:
            x = self.resample(x)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        if (h.shape[1] & (h.shape[1] - 1) == 0) and h.shape[1] > 0:
            h = self.wht(h)
        h = self.conv2(F.silu(self.norm2(h)))
        h = self.se(h)
        return h + self.skip(x)


# ================================================================
# SPADE
# ================================================================

class SPADEGroupNorm(nn.Module):
    def __init__(self, num_channels, cond_channels):
        super().__init__()
        g = _safe_groups(num_channels)
        self.norm   = nn.GroupNorm(g, num_channels, affine=False)
        hidden      = max(64, num_channels // 2)
        self.shared = nn.Sequential(
            nn.Conv2d(cond_channels, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.gamma = nn.Conv2d(hidden, num_channels, 3, padding=1)
        self.beta  = nn.Conv2d(hidden, num_channels, 3, padding=1)
        self.wht   = WHTLayer()
        nn.init.zeros_(self.gamma.bias);  nn.init.normal_(self.gamma.weight, 0, 0.02)
        nn.init.zeros_(self.beta.bias);   nn.init.normal_(self.beta.weight,  0, 0.02)

    def forward(self, x, cond):
        norm_x    = self.norm(x)
        cond_feat = self.wht(self.shared(cond))
        return norm_x * (1.0 + self.gamma(cond_feat)) + self.beta(cond_feat)


# ================================================================
# ENCODER / DECODER BLOCKS
# ================================================================

class DualEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        g = _safe_groups(out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch), nn.SiLU(),
            SEBlock(out_ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch), nn.SiLU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


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
# CORR DIFF REGRESSOR (Stage 1 — original, unchanged)
# ================================================================

class CorrDiffRegressor(nn.Module):
    """
    Dual-encoder U-Net for 1deg -> 0.25deg rainfall downscaling.
    (Original architecture — kept unchanged for backward compatibility.)
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

        stem_ch = base_channels // 2   # 32

        self.rain_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, stem_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(stem_ch), stem_ch), nn.SiLU(),
        )
        self.topo_stem = nn.Sequential(
            nn.Conv2d(1, stem_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(stem_ch), stem_ch), nn.SiLU(),
        )
        self.global_stem_proj = (nn.Linear(emb_dim, stem_ch)
                                  if self.global_mlp else None)

        self.rain_enc    = nn.ModuleList()
        self.topo_enc    = nn.ModuleList()
        self.rain_down   = nn.ModuleList()
        self.topo_down   = nn.ModuleList()
        self.enc_skip_ch = []

        curr_rain_ch = curr_topo_ch = stem_ch
        for i, mult in enumerate(self.channel_mult):
            out_ch   = base_channels * mult
            r_blocks = nn.ModuleList()
            t_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                r_blocks.append(DualEncoderBlock(curr_rain_ch, out_ch, dropout))
                t_blocks.append(DualEncoderBlock(curr_topo_ch, out_ch, dropout))
                curr_rain_ch = curr_topo_ch = out_ch
            self.rain_enc.append(r_blocks)
            self.topo_enc.append(t_blocks)
            self.enc_skip_ch.append(curr_rain_ch + curr_topo_ch)
            if i != len(self.channel_mult) - 1:
                self.rain_down.append(nn.Conv2d(curr_rain_ch, curr_rain_ch, 4, stride=2, padding=1))
                self.topo_down.append(nn.Conv2d(curr_topo_ch, curr_topo_ch, 4, stride=2, padding=1))
            else:
                self.rain_down.append(None)
                self.topo_down.append(None)

        bn_in = curr_rain_ch + curr_topo_ch
        bn_ch = base_channels * self.channel_mult[-1]
        self.bn_proj   = nn.Conv2d(bn_in, bn_ch, 1)
        self.wht_heal  = WHTLayer()
        self.bn_attn   = nn.MultiheadAttention(bn_ch, max(1, bn_ch // 64), batch_first=True)
        self.bn_mid    = nn.Sequential(
            nn.GroupNorm(_safe_groups(bn_ch), bn_ch), nn.SiLU(),
            nn.Conv2d(bn_ch, bn_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(bn_ch), bn_ch), nn.SiLU(),
        )
        self.global_bn_proj = (nn.Linear(emb_dim, bn_ch) if self.global_mlp else None)

        COND_CH = 2
        self.dec_ups    = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_ch          = bn_ch

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
        self.final_conv = nn.Conv2d(dec_ch, out_channels, 3, padding=1)
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, topo, global_features=None):
        g_emb = self.global_mlp(global_features) if (self.global_mlp and global_features is not None) else None

        rain = self.rain_upsample(x)
        tf   = self.topo_stem(topo)

        if g_emb is not None and self.global_stem_proj is not None:
            g_stem = self.global_stem_proj(g_emb)[:, :, None, None]
            rain = rain + g_stem
            tf   = tf   + g_stem

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

        fused = self.wht_heal(torch.cat([rain, tf], dim=1))
        fused = self.bn_proj(fused)
        if g_emb is not None and self.global_bn_proj is not None:
            fused = fused + self.global_bn_proj(g_emb)[:, :, None, None]
        B, C, H, W = fused.shape
        flat = fused.view(B, C, -1).permute(0, 2, 1)
        attn_out, _ = self.bn_attn(flat, flat, flat)
        fused = fused + attn_out.permute(0, 2, 1).view(B, C, H, W)
        fused = self.bn_mid(fused)

        x_dec = fused
        for i, (up, blk_list) in enumerate(zip(self.dec_ups, self.dec_blocks)):
            level = len(self.channel_mult) - 1 - i
            if up is not None:
                x_dec = up(x_dec)
            skip  = torch.cat([rain_skips[level], topo_skips[level]], dim=1)
            x_dec = torch.cat([x_dec, skip], dim=1)
            cond_r = F.interpolate(cond_fine, size=x_dec.shape[-2:],
                                   mode='bilinear', align_corners=False)
            for blk in blk_list:
                x_dec = blk(x_dec, cond_r)

        return self.final_conv(F.silu(self.final_norm(x_dec)))


# ================================================================
# TEMPORAL CORRDIFF REGRESSOR  (NEW — Stage 1 with history)
# ================================================================

class TemporalCorrDiffRegressor(nn.Module):
    """
    CorrDiffRegressor extended with a ConvLSTM temporal encoder.

    New argument in forward():
        seq  : [B, T, in_channels, H_coarse, W_coarse]
               — T previous coarse frames (not including the current one)
               — if None, behaves identically to CorrDiffRegressor

    When seq is provided:
        1. TemporalEncoder processes the sequence → context [B, stem_ch, H_f, W_f]
        2. context is added to the rain-stem output after upsampling
        3. Everything else is the same as CorrDiffRegressor

    This means:
        - No extra training data format change is required (seq can be zeros
          for the first time-step or for datasets with no history)
        - The model degrades gracefully to zero-context when seq is None

    Constructor args are identical to CorrDiffRegressor plus:
        seq_len      : number of historical timesteps the LSTM expects
        lstm_layers  : ConvLSTM depth (2 is usually enough)
        lstm_drop    : inter-layer dropout inside the LSTM
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
        seq_len=5,          # number of historical coarse frames
        lstm_layers=2,
        lstm_drop=0.0,
        fine_size=128,      # output spatial resolution
        **kwargs
    ):
        super().__init__()
        self.seq_len    = seq_len

        # ---- Core regressor (all the same weights) ----
        self.core = CorrDiffRegressor(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_blocks=num_blocks,
            global_dim=global_dim,
            dropout=dropout,
        )

        stem_ch = base_channels // 2   # must match core's stem_ch

        # ---- Temporal encoder ----
        self.temporal_enc = TemporalEncoder(
            in_ch=in_channels,
            stem_ch=stem_ch,
            lstm_layers=lstm_layers,
            lstm_drop=lstm_drop,
            fine_size=fine_size,
        )

        # ---- Gating: blend temporal context into rain stem ----
        # A learned gate prevents the temporal context from dominating
        # early in training before the LSTM has converged.
        self.ctx_gate = nn.Sequential(
            nn.Conv2d(stem_ch * 2, stem_ch, 1),   # cat(rain_stem, ctx) → gate weight
            nn.Sigmoid(),
        )

    def forward(self, x, topo, global_features=None, seq=None):
        """
        x    : [B, in_ch, H_c, W_c]       — current coarse frame
        topo : [B, 1,     H_f, W_f]        — fine topography
        seq  : [B, T, in_ch, H_c, W_c]    — optional historical coarse frames
        """
        # If no history provided, delegate to core directly
        if seq is None:
            return self.core(x, topo, global_features)

        # Compute temporal context
        ctx = self.temporal_enc(seq)               # [B, stem_ch, H_f, W_f]

        # We need to inject ctx right after the rain stem.
        # Monkey-patching forward is messy; instead we hook into the stem.
        # We run the stem ourselves then add the gated context.
        g_emb = (self.core.global_mlp(global_features)
                 if (self.core.global_mlp and global_features is not None)
                 else None)

        rain = self.core.rain_upsample(x)          # [B, stem_ch, H_f, W_f]
        tf   = self.core.topo_stem(topo)

        if g_emb is not None and self.core.global_stem_proj is not None:
            g_stem = self.core.global_stem_proj(g_emb)[:, :, None, None]
            rain   = rain + g_stem
            tf     = tf   + g_stem

        # Gated fusion of temporal context into rain features
        gate   = self.ctx_gate(torch.cat([rain, ctx], dim=1))   # [B, stem_ch, H, W]
        rain   = rain + gate * ctx                               # residual addition

        # ---------- from here: replicate core.forward internals ----------
        clim_fine = F.interpolate(x[:, 0:1], size=topo.shape[-2:],
                                  mode='bilinear', align_corners=False).detach()
        cond_fine  = torch.cat([clim_fine, topo], dim=1)

        rain_skips, topo_skips = [], []
        for i in range(len(self.core.channel_mult)):
            for r_blk, t_blk in zip(self.core.rain_enc[i], self.core.topo_enc[i]):
                rain = r_blk(rain); tf = t_blk(tf)
            rain_skips.append(rain); topo_skips.append(tf)
            if self.core.rain_down[i] is not None:
                rain = self.core.rain_down[i](rain)
                tf   = self.core.topo_down[i](tf)

        fused = self.core.wht_heal(torch.cat([rain, tf], dim=1))
        fused = self.core.bn_proj(fused)
        if g_emb is not None and self.core.global_bn_proj is not None:
            fused = fused + self.core.global_bn_proj(g_emb)[:, :, None, None]
        B, C, H, W = fused.shape
        flat = fused.view(B, C, -1).permute(0, 2, 1)
        attn_out, _ = self.core.bn_attn(flat, flat, flat)
        fused = fused + attn_out.permute(0, 2, 1).view(B, C, H, W)
        fused = self.core.bn_mid(fused)

        x_dec = fused
        for i, (up, blk_list) in enumerate(zip(self.core.dec_ups, self.core.dec_blocks)):
            level = len(self.core.channel_mult) - 1 - i
            if up is not None:
                x_dec = up(x_dec)
            skip  = torch.cat([rain_skips[level], topo_skips[level]], dim=1)
            x_dec = torch.cat([x_dec, skip], dim=1)
            cond_r = F.interpolate(cond_fine, size=x_dec.shape[-2:],
                                   mode='bilinear', align_corners=False)
            for blk in blk_list:
                x_dec = blk(x_dec, cond_r)

        return self.core.final_conv(F.silu(self.core.final_norm(x_dec)))


# ================================================================
# WEIGHTED ENSEMBLE  (NEW — inference only)
# ================================================================

class WeightedEnsemble(nn.Module):
    """
    Load N checkpoint files and combine their predictions with learned or
    validation-based weights.

    Usage:
        ens = WeightedEnsemble(
            model_cls   = CorrDiffRegressor,
            model_kwargs= dict(in_channels=2, out_channels=1),
            ckpt_paths  = ["best_seed0.pth", "best_seed1.pth", ...],
            weights     = [0.8, 0.6, ...]   # validation wPCC scores
        )
        ens.eval()
        with torch.no_grad():
            pred = ens(xi, topo=topo, global_features=tf)

    Args:
        model_cls    : class to instantiate (CorrDiffRegressor or TemporalCorrDiffRegressor)
        model_kwargs : dict passed to model_cls constructor
        ckpt_paths   : list of checkpoint paths
        weights      : list of raw validation scores (wPCC); will be softmax-normalised
                       if None, uniform weights are used
        ema          : if True, load "ema_state_dict" from checkpoint (default True)
        device       : torch.device
    """
    def __init__(
        self,
        model_cls,
        model_kwargs: dict,
        ckpt_paths:   List[str],
        weights:      Optional[List[float]] = None,
        ema:          bool = True,
        device:       Optional[torch.device] = None,
    ):
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
            for p in m.parameters():
                p.requires_grad_(False)
            self.models.append(m)

        # Normalise weights via softmax so they sum to 1
        if weights is None:
            w = torch.ones(len(ckpt_paths), device=device) / len(ckpt_paths)
        else:
            raw = torch.tensor(weights, dtype=torch.float32, device=device)
            # Temperature-scaled softmax — temperature=0.5 sharpens toward best model
            w = torch.softmax(raw / 0.5, dim=0)
        # Register as buffer so .to(device) works
        self.register_buffer("weights", w)

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Weighted average of all member predictions.
        All *args and **kwargs are forwarded to each model.
        """
        preds = []
        for m in self.models:
            preds.append(m(*args, **kwargs))
        stacked = torch.stack(preds, dim=0)        # [N, B, C, H, W]
        w       = self.weights.view(-1, 1, 1, 1, 1)
        return (stacked * w).sum(dim=0)            # [B, C, H, W]


def ensemble_predict(
    model_cls,
    model_kwargs: dict,
    ckpt_paths:   List[str],
    xi:           torch.Tensor,
    topo:         torch.Tensor,
    tf:           torch.Tensor,
    weights:      Optional[List[float]] = None,
    device:       Optional[torch.device] = None,
    seq:          Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Convenience wrapper: build ensemble on the fly, run one forward pass,
    return prediction [B, 1, H, W] in log1p space.

    This function is mainly for evaluation scripts.  For training-time
    ensemble selection, use WeightedEnsemble directly.
    """
    ens = WeightedEnsemble(model_cls, model_kwargs, ckpt_paths, weights, device=device)
    ens.eval()
    kwargs = dict(topo=topo, global_features=tf)
    if seq is not None:
        kwargs["seq"] = seq
    with torch.no_grad():
        return ens(xi, **kwargs)


# ================================================================
# UNET FOR DIFFUSION (Stage 2 — original, unchanged)
# ================================================================

class UNet(nn.Module):
    """
    Base U-Net architecture for diffusion denoiser.
    (Original, unchanged — Stage-2 training script requires no modification.)
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

        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(base_channels),
            nn.Linear(base_channels, emb_ch), nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )
        self.global_mlp = nn.Linear(global_dim, emb_ch) if global_dim else None
        self.head       = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down = nn.ModuleList()
        self.up   = nn.ModuleList()
        curr_ch   = base_channels
        skip_channels = []

        for mult in channel_mult:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(curr_ch, out_ch, emb_ch))
                curr_ch = out_ch
                skip_channels.append(curr_ch)
            self.down.append(ResBlock(curr_ch, curr_ch, emb_ch, down=True))
            skip_channels.append(curr_ch)

        self.mid = nn.ModuleList([
            ResBlock(curr_ch, curr_ch, emb_ch),
            ResBlock(curr_ch, curr_ch, emb_ch),
        ])

        self.use_bottleneck_attention = use_bottleneck_attention
        if use_bottleneck_attention:
            self.bn_attn = nn.MultiheadAttention(curr_ch, max(1, curr_ch // 64), batch_first=True)

        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            self.up.append(ResBlock(curr_ch + skip_channels.pop(), out_ch, emb_ch, up=True))
            curr_ch = out_ch
            for _ in range(num_res_blocks):
                self.up.append(ResBlock(curr_ch + skip_channels.pop(), out_ch, emb_ch))
                curr_ch = out_ch

        self.final = nn.Sequential(
            nn.GroupNorm(_safe_groups(curr_ch), curr_ch),
            nn.SiLU(),
            nn.Conv2d(curr_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, t, global_features=None):
        emb = self.time_mlp(t)
        if global_features is not None and self.global_mlp is not None:
            emb = emb + self.global_mlp(global_features)

        h     = self.head(x)
        skips = [h]

        for layer in self.down:
            h = layer(h, emb)
            skips.append(h)

        for layer in self.mid:
            h = layer(h, emb)

        if self.use_bottleneck_attention:
            B, C, H, W = h.shape
            flat      = h.view(B, C, -1).permute(0, 2, 1)
            flat_norm = F.layer_norm(flat, [flat.size(-1)])
            attn_out, _ = self.bn_attn(flat_norm, flat_norm, flat_norm)
            h = h + attn_out.permute(0, 2, 1).view(B, C, H, W)

        for layer in self.up:
            skip = skips.pop()
            h    = torch.cat([h, skip], dim=1)
            h    = layer(h, emb)

        return self.final(h)
