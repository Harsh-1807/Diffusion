import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Utilities
# ============================================================

def group_norm(ch):
    for g in [32,16,8,4,2,1]:
        if ch % g == 0:
            return nn.GroupNorm(g,ch)
    return nn.GroupNorm(1,ch)


# ============================================================
# Time embedding
# ============================================================

class SinusoidalEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):

        half = self.dim // 2

        emb = math.log(10000) / (half - 1)

        emb = torch.exp(torch.arange(half, device=t.device) * -emb)

        emb = t[:,None] * emb[None,:]

        return torch.cat([emb.sin(), emb.cos()], dim=1)


# ============================================================
# Residual Block
# ============================================================

class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, emb_ch):
        super().__init__()

        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch,out_ch,3,padding=1)

        self.emb_proj = nn.Linear(emb_ch,out_ch)

        self.norm2 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch,out_ch,3,padding=1)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch,out_ch,1)
        else:
            self.skip = nn.Identity()

    def forward(self,x,emb):

        h = self.conv1(F.silu(self.norm1(x)))

        emb_out = self.emb_proj(F.silu(emb))[:,:,None,None]

        h = h + emb_out

        h = self.conv2(F.silu(self.norm2(h)))

        return h + self.skip(x)


# ============================================================
# Downsample
# ============================================================

class Down(nn.Module):

    def __init__(self,ch):
        super().__init__()
        self.conv = nn.Conv2d(ch,ch,4,stride=2,padding=1)

    def forward(self,x):
        return self.conv(x)


# ============================================================
# Upsample
# ============================================================

class Up(nn.Module):

    def __init__(self,ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch,ch,4,stride=2,padding=1)

    def forward(self,x):
        return self.conv(x)


# ============================================================
# Diffusion UNet
# ============================================================

class DiffusionUNet(nn.Module):

    def __init__(
        self,
        in_channels=12,
        out_channels=4,
        base_channels=64
    ):

        super().__init__()

        emb_ch = base_channels*4

        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(base_channels),
            nn.Linear(base_channels,emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch,emb_ch)
        )

        self.head = nn.Conv2d(in_channels,base_channels,3,padding=1)


        # encoder
        self.enc1 = ResBlock(base_channels,64,emb_ch)
        self.enc2 = ResBlock(64,128,emb_ch)
        self.enc3 = ResBlock(128,256,emb_ch)

        self.down1 = Down(64)
        self.down2 = Down(128)
        self.down3 = Down(256)


        # bottleneck
        self.mid1 = ResBlock(256,256,emb_ch)
        self.mid2 = ResBlock(256,256,emb_ch)


        # decoder
        self.up3 = Up(256)
        self.dec3 = ResBlock(512,128,emb_ch)

        self.up2 = Up(128)
        self.dec2 = ResBlock(256,64,emb_ch)

        self.up1 = Up(64)
        self.dec1 = ResBlock(128,64,emb_ch)


        self.final = nn.Sequential(
            group_norm(64),
            nn.SiLU(),
            nn.Conv2d(64,out_channels,3,padding=1)
        )


    def forward(self,x,t):

        emb = self.time_mlp(t)

        h = self.head(x)

        s1 = self.enc1(h,emb)
        h = self.down1(s1)

        s2 = self.enc2(h,emb)
        h = self.down2(s2)

        s3 = self.enc3(h,emb)
        h = self.down3(s3)


        h = self.mid1(h,emb)
        h = self.mid2(h,emb)


        h = self.up3(h)

        if s3.shape[-2:] != h.shape[-2:]:
            s3 = F.interpolate(s3,size=h.shape[-2:],mode="bilinear",align_corners=False)

        h = self.dec3(torch.cat([h,s3],dim=1),emb)


        h = self.up2(h)

        if s2.shape[-2:] != h.shape[-2:]:
            s2 = F.interpolate(s2,size=h.shape[-2:],mode="bilinear",align_corners=False)

        h = self.dec2(torch.cat([h,s2],dim=1),emb)


        h = self.up1(h)

        if s1.shape[-2:] != h.shape[-2:]:
            s1 = F.interpolate(s1,size=h.shape[-2:],mode="bilinear",align_corners=False)

        h = self.dec1(torch.cat([h,s1],dim=1),emb)


        return self.final(h)
