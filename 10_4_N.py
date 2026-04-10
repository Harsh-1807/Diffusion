# -*- coding: utf-8 -*-
"""
CorrDiff Network Architecture (Stage 1 & 2)
============================================

CorrDiffRegressor (Stage 1): Deterministic 4x super-resolution
  - Dual-encoder: coarse+clim_bias | topo (merged at bottleneck)
  - SPADE-conditioned decoder: clim_bias+topo modulate every level
  - Output: Fine-resolution rainfall prediction

UNet (Stage 2): Stochastic diffusion denoiser base class
  - Conditions on regressor output + coarse input + topo
  - Samples realistic variability around regressor mean

Key fixes in this version:
  [1] final_conv: kaiming_normal_ init (NOT zeros)
  [2] global_stem_proj, global_bn_proj: proper Linear layers (NOT slicing)
  [3] SPADE in decoder: fine-resolution conditioning at every level
  [4] UNet.forward: removed duplicate emb computation, fixed g -> global_features
  [5] UNet: bottleneck attention applied after self.mid (NOT before encoder)
  [6] UNet: num_blocks alias accepted alongside num_res_blocks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Sinusoidal positional embeddings (for time in diffusion models)."""
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
    """Squeeze-and-excitation block for channel-wise attention."""
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
    """The 'Healing Balm' - Fast Walsh-Hadamard Transform."""
    def __init__(self, scale=True):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        # x: [B, C, H, W] or [B, N, C]
        return self.apply_wht(x)

    def apply_wht(self, x):
        """Recursive or iterative FWHT implementation."""
        # Note: In practice, use a vectorized FWHT for PyTorch. 
        # For small bottleneck dimensions, a fixed Hadamard matrix works too.
        n = x.shape[1] # Assumes power of 2
        if n == 1: return x
        x = x.view(-1, 2, n//2, *x.shape[2:])
        a, b = x[:, 0], x[:, 1]
        res = torch.cat([a + b, a - b], dim=1)
        if self.scale:
            res = res / math.sqrt(2)
        return res

# ================================================================
# RESIDUAL BLOCK (for diffusion denoiser)
# ================================================================

class ResBlock(nn.Module):
    """Residual block with 'Healing Balm' (WHT) between convolutions."""
    def __init__(self, in_ch, out_ch, emb_ch, down=False, up=False):
        super().__init__()
        self.down = down
        self.up = up
        self.se = SEBlock(out_ch)
        self.wht = WHTLayer() # The Balm

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
            
        # --- First Stage: Local Extraction ---
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        
        # --- The Balm: Channel Diffusion ---
        # Only apply if channels are a power of 2 to avoid crashes
        if (h.shape[1] & (h.shape[1] - 1) == 0) and h.shape[1] > 0:
            h = self.wht(h)
            
        # --- Second Stage: Refinement ---
        h = self.conv2(F.silu(self.norm2(h)))
        h = self.se(h)
        return h + self.skip(x)


# ================================================================
# SPADE: Spatially-Adaptive Denormalization
# ================================================================

class SPADEGroupNorm(nn.Module):
    """
    SPADE (Spatially-Adaptive Denormalization) for decoder.

    Replaces standard affine GroupNorm with spatially-varying gamma/beta
    predicted from a conditioning map (clim_bias + topo).

    WHY: Standard GN applies a single scalar per channel.
         SPADE applies different scale/shift at every pixel,
         directly guided by the conditioning map.
         This lets the decoder amplify features at rain hotspots
         and suppress them over dry plains.

    INIT: gamma and beta start at 0, making this an identity transform initially.
          This ensures stable training in early epochs.
    """
    def __init__(self, num_channels, cond_channels):
        super().__init__()
        g = _safe_groups(num_channels)
        self.norm = nn.GroupNorm(g, num_channels, affine=False)

        hidden = max(64, num_channels // 2)
        self.shared = nn.Sequential(
            nn.Conv2d(cond_channels, hidden, 3, padding=1),
            nn.SiLU(),
        )

        self.gamma = nn.Conv2d(hidden, num_channels, 3, padding=1)
        self.beta  = nn.Conv2d(hidden, num_channels, 3, padding=1)
        self.wht = WHTLayer()

        # Identity init: (1 + 0) * norm_x + 0 = norm_x
        
        nn.init.zeros_(self.gamma.bias)
        nn.init.normal_(self.gamma.weight, 0, 0.02)
        nn.init.normal_(self.beta.weight, 0, 0.02)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, cond):
        norm_x = self.norm(x)
        cond_feat = self.shared(cond)
        
        # Heal the conditioning feature spatial-channel-wise
        # We treat the channel dimension as the vector to be diffused
        cond_feat = self.wht(cond_feat) 
        
        return norm_x * (1.0 + self.gamma(cond_feat)) + self.beta(cond_feat)


# ================================================================
# DUAL ENCODER BLOCK (standard, no SPADE)
# ================================================================

class DualEncoderBlock(nn.Module):
    """
    Standard encoder block with GroupNorm (no SPADE).

    Encoder should NOT see fine-resolution conditioning maps
    (clim_bias, topo) to avoid information leakage.
    """
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        g = _safe_groups(out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(),
            SEBlock(out_ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


# ================================================================
# SPADE DECODER BLOCK (SPADE-conditioned)
# ================================================================

class SPADEDecoderBlock(nn.Module):
    """
    Decoder block with SPADE-conditioned normalization.

    Uses SPADEGroupNorm so clim_bias + topo directly modulate
    spatial feature activations at every decoder level.
    """
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
        h = self.norm1(self.conv1(x), cond)
        h = F.silu(h)
        h = self.drop(h)
        h = self.norm2(self.conv2(h), cond)
        h = self.se(F.silu(h))
        return h + self.skip(x)


# ================================================================
# CORR DIFF REGRESSOR (Stage 1)
# ================================================================

class CorrDiffRegressor(nn.Module):
    """
    Dual-encoder U-Net for 1deg -> 0.25deg rainfall downscaling.

    Architecture:
      1. Rain stem: coarse [B,2,32,32] -> [B,stem_ch,128,128] via 4x ConvTranspose
      2. Topo stem: [B,1,128,128] -> [B,stem_ch,128,128]
      3. Dual encoders: separate streams for rain | topo, merged at bottleneck
      4. Bottleneck: self-attention + global feature injection
      5. Decoder: SPADE-conditioned blocks with skip connections
      6. Output: [B,1,128,128] fine rainfall via final_conv
    """
    def __init__(
        self,
        in_channels,      # 2: [coarse, clim_bias_coarse]
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

        # ---- Global feature embedding ----
        if global_dim is not None and global_dim > 0:
            self.global_mlp = nn.Sequential(
                nn.Linear(global_dim, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim),
                nn.SiLU(),
            )
        else:
            self.global_mlp = None

        stem_ch = base_channels // 2  # 32

        # ---- Rain stem: 4x upsampling ----
        # Replace the rain_upsample with a more stable PixelShuffle or Bilinear + Conv
        self.rain_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, stem_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(stem_ch), stem_ch),
            nn.SiLU()
        )

        # ---- Topo stem ----
        self.topo_stem = nn.Sequential(
            nn.Conv2d(1, stem_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(stem_ch), stem_ch),
            nn.SiLU(),
        )

        if self.global_mlp is not None:
            self.global_stem_proj = nn.Linear(emb_dim, stem_ch)
        else:
            self.global_stem_proj = None

        # ---- Dual encoder streams ----
        self.rain_enc    = nn.ModuleList()
        self.topo_enc    = nn.ModuleList()
        self.rain_down   = nn.ModuleList()
        self.topo_down   = nn.ModuleList()
        self.enc_skip_ch = []

        curr_rain_ch = stem_ch
        curr_topo_ch = stem_ch

        for i, mult in enumerate(self.channel_mult):
            out_ch   = base_channels * mult
            r_blocks = nn.ModuleList()
            t_blocks = nn.ModuleList()

            for _ in range(num_blocks):
                r_blocks.append(DualEncoderBlock(curr_rain_ch, out_ch, dropout))
                t_blocks.append(DualEncoderBlock(curr_topo_ch, out_ch, dropout))
                curr_rain_ch = out_ch
                curr_topo_ch = out_ch

            self.rain_enc.append(r_blocks)
            self.topo_enc.append(t_blocks)
            self.enc_skip_ch.append(curr_rain_ch + curr_topo_ch)

            if i != len(self.channel_mult) - 1:
                self.rain_down.append(
                    nn.Conv2d(curr_rain_ch, curr_rain_ch, 4, stride=2, padding=1))
                self.topo_down.append(
                    nn.Conv2d(curr_topo_ch, curr_topo_ch, 4, stride=2, padding=1))
            else:
                self.rain_down.append(None)
                self.topo_down.append(None)

        # ---- Bottleneck ----
        bn_in = curr_rain_ch + curr_topo_ch
        bn_ch = base_channels * self.channel_mult[-1]
        self.bn_proj = nn.Conv2d(bn_in, bn_ch, 1)
        self.wht_heal = WHTLayer()
        self.bn_attn = nn.MultiheadAttention(
            embed_dim=bn_ch, num_heads=max(1, bn_ch // 64), batch_first=True
        )
        self.bn_mid = nn.Sequential(
            nn.GroupNorm(_safe_groups(bn_ch), bn_ch),
            nn.SiLU(),
            nn.Conv2d(bn_ch, bn_ch, 3, padding=1),
            nn.GroupNorm(_safe_groups(bn_ch), bn_ch),
            nn.SiLU(),
        )

        if self.global_mlp is not None:
            self.global_bn_proj = nn.Linear(emb_dim, bn_ch)
        else:
            self.global_bn_proj = None

        # ---- SPADE conditioning channels ----
        COND_CH = 2  # [clim_bias, topo]

        # ---- Decoder ----
        self.dec_ups    = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_ch          = bn_ch

        for i, mult in reversed(list(enumerate(self.channel_mult))):
            out_ch  = base_channels * mult
            skip_ch = self.enc_skip_ch[i]

            if i != len(self.channel_mult) - 1:
                self.dec_ups.append(
                    nn.ConvTranspose2d(dec_ch, dec_ch, 4, stride=2, padding=1))
            else:
                self.dec_ups.append(None)

            dec_in   = dec_ch + skip_ch
            blk_list = nn.ModuleList()
            for _ in range(num_blocks):
                blk_list.append(SPADEDecoderBlock(dec_in, out_ch, COND_CH, dropout))
                dec_in = out_ch

            self.dec_blocks.append(blk_list)
            dec_ch = out_ch

        # ---- Final output ----
        self.final_norm = nn.GroupNorm(_safe_groups(dec_ch), dec_ch)
        self.final_conv = nn.Conv2d(dec_ch, out_channels, 3, padding=1)
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, topo, global_features=None):
        # ---- Global embedding ----
        g_emb = None
        if self.global_mlp is not None and global_features is not None:
            g_emb = self.global_mlp(global_features)

        # ---- Stems ----
        rain = self.rain_upsample(x)
        tf   = self.topo_stem(topo)

        if g_emb is not None and self.global_stem_proj is not None:
            g_stem = self.global_stem_proj(g_emb)[:, :, None, None]
            rain   = rain + g_stem
            tf     = tf   + g_stem

        # ---- SPADE conditioning map ----
        clim_fine = F.interpolate(
            x[:, 0:1], size=topo.shape[-2:],
            mode='bilinear', align_corners=False
        ).detach()
        cond_fine = torch.cat([clim_fine, topo], dim=1)  # [B, 2, H, W]

        # ---- Encoder ----
        rain_skips, topo_skips = [], []
        for i in range(len(self.channel_mult)):
            for r_blk, t_blk in zip(self.rain_enc[i], self.topo_enc[i]):
                rain = r_blk(rain)
                tf   = t_blk(tf)
            rain_skips.append(rain)
            topo_skips.append(tf)
            if self.rain_down[i] is not None:
                rain = self.rain_down[i](rain)
                tf   = self.topo_down[i](tf)

        # ---- Bottleneck (Corrected Order) ----
        # 1. Cat the encoder outputs
        fused = torch.cat([rain, tf], dim=1) 
        
        # 2. Heal the "Fracturing" 
        if hasattr(self, 'wht_heal'):
            fused = self.wht_heal(fused) 
        
        # 3. Project to bottleneck dimension
        fused = self.bn_proj(fused) 
        
        # 4. Global feature injection
        if g_emb is not None and self.global_bn_proj is not None:
            g_bn = self.global_bn_proj(g_emb)[:, :, None, None]
            fused = fused + g_bn
        
        # 5. Attention
        B, C, H, W = fused.shape
        flat = fused.view(B, C, -1).permute(0, 2, 1)
        attn_out, _ = self.bn_attn(flat, flat, flat)
        fused = fused + attn_out.permute(0, 2, 1).view(B, C, H, W)
        
        # 6. Final mid-blocks
        fused = self.bn_mid(fused)

        # ---- Decoder (SPADE-conditioned) ----
        x_dec = fused
        for i, (up, blk_list) in enumerate(zip(self.dec_ups, self.dec_blocks)):
            level = len(self.channel_mult) - 1 - i
            if up is not None:
                x_dec = up(x_dec)

            skip  = torch.cat([rain_skips[level], topo_skips[level]], dim=1)
            x_dec = torch.cat([x_dec, skip], dim=1)

            cond_resized = F.interpolate(
                cond_fine, size=x_dec.shape[-2:],
                mode='bilinear', align_corners=False
            )
            for blk in blk_list:
                x_dec = blk(x_dec, cond_resized)

        out = self.final_conv(F.silu(self.final_norm(x_dec)))
        return out # Note: The softplus is applied in TrainRegressor.py, so we return raw here.

# ================================================================
# UNET FOR DIFFUSION (Stage 2)
# ================================================================

class UNet(nn.Module):
    """
    Base U-Net architecture for diffusion denoiser.

    Fixes vs previous version:
      [1] forward(): removed duplicate emb computation
      [2] forward(): g -> global_features (consistent naming)
      [3] bottleneck attention applied after self.mid, NOT before encoder
      [4] num_blocks accepted as alias for num_res_blocks
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        num_blocks=None,                  # alias for num_res_blocks
        global_dim=2,
        use_bottleneck_attention=False,
        **kwargs
    ):
        super().__init__()

        # num_blocks takes priority if provided
        num_res_blocks = num_blocks if num_blocks is not None else num_res_blocks

        emb_ch = base_channels * 4

        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(base_channels),
            nn.Linear(base_channels, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch)
        )

        self.global_mlp = nn.Linear(global_dim, emb_ch) if global_dim else None
        self.head       = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down = nn.ModuleList()
        self.up   = nn.ModuleList()

        curr_ch       = base_channels
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

        # FIX: bottleneck attention lives here, applied after self.mid in forward()
        self.use_bottleneck_attention = use_bottleneck_attention
        if use_bottleneck_attention:
            self.bn_attn = nn.MultiheadAttention(
                embed_dim=curr_ch, num_heads=max(1, curr_ch // 64), batch_first=True
            )

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
            nn.Conv2d(curr_ch, out_channels, 3, padding=1)
        )

    def forward(self, x, t, global_features=None):
        # FIX [1+2]: single emb computation, correct variable name
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

        # FIX [3]: bottleneck attention applied HERE, after mid blocks
        if self.use_bottleneck_attention:
            B, C, H, W = h.shape
            flat = h.view(B, C, -1).permute(0, 2, 1)
            # Add a LayerNorm before attention for stability
            flat_norm = F.layer_norm(flat, [flat.size(-1)]) 
            attn_out, _ = self.bn_attn(flat_norm, flat_norm, flat_norm)
            h = h + attn_out.permute(0, 2, 1).view(B, C, H, W)

        for layer in self.up:
            skip = skips.pop()
            h    = torch.cat([h, skip], dim=1)
            h    = layer(h, emb)

        return self.final(h)
