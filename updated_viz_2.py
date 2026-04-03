# -*- coding: utf-8 -*-
# EvalCorrDiff_Singapore_v4_fixed_v2.py
#
# ROOT CAUSE FIXES:
#   [R1]  Embedding key mismatch: training saves "embedding", eval looked for
#         "embedding_state". Now checks both keys. Without this, embedding_fw
#         is always None and the model sees completely different conditioning
#         than during training -> near-zero precipitation predictions.
#   [R2]  use_embedding flag: training checkpoint has no "use_embedding" key,
#         so the old guard `if use_emb and emb_state` always failed. Now we
#         just check if the embedding dict is present and fitted.
#   [R3]  EmbeddingFramework.load_state_dict: training state_dict has
#         "qdm_lr_local"/"qdm_hr_local" keys that the eval version didn't
#         handle -> KeyError on load. Added safe .get() fallback.
#   [F1]  make_latlon: uses actual HR field shape, not hardcoded (256, 276).
#   [F2]  to_mm: clips to >= 0 (precipitation cannot be negative).
#   [F3]  Lat orientation: determined once from raw dataset lat; all fields
#         flipped consistently before clipping.

import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import geopandas as gpd
import xarray as xr
import pandas as pd

from Dataset import ClimateDataset
from Network import DiffusionUNet
from Regressor import ClimateRegressor
from shapely.geometry import Point

# =============================================================================
# PATHS
# =============================================================================

LR_FILES = [
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/huss_rcm_8km_daily_data_1995-2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/mslp_rcm_8km_daily_data_1995-2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/tas_rcm_8km_daily_data_1995-2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/precip_rcm_8km_daily_data_1995-2014.nc",
]

BEST_CKPT = ("/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/CorrDiff/version11/corrdiff_singapore_v7_best.pth")
REG_CKPT  = "regressor.pth"
SHAPEFILE = ("/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/"
             "ShapeFiles/singapore_Singapore_Country_Boundary.shp")

WORLD_COUNTRIES_SHP = ("/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/"
                       "ShapeFiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp")

OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRECIP_CH   = 3
N_CH        = 4
N_QUANTILES = 500
N_INFER     = 50
_RHO        = 5.0

CENTER_LAT = 1.3
CENTER_LON = 103.8
HALF_SIZE  = 0.5

SG_LON_MIN = CENTER_LON - HALF_SIZE
SG_LON_MAX = CENTER_LON + HALF_SIZE
SG_LAT_MIN = CENTER_LAT - HALF_SIZE
SG_LAT_MAX = CENTER_LAT + HALF_SIZE

# Singapore Gaussian weight map params (must match training)
SG_LAT_C  = 1.3
SG_LON_C  = 103.8
SG_SIGMA  = 1.2
SG_WEIGHT = 11.0

TEXT  = "black"
BG    = "white"
GRID  = "gray"
SUB   = "black"
PANEL = "white"


# =============================================================================
# COLOURMAP
# =============================================================================
def make_dynamic_cmap(vmax, n_levels=9, kind="rain", vmin=None):
    vmax = max(float(vmax), 1e-6)
    if kind == "rain":
        raw_step  = vmax / max(n_levels - 1, 1)
        magnitude = 10 ** math.floor(math.log10(max(raw_step, 1e-9)))
        step      = magnitude * round(raw_step / magnitude)
        if step <= 0:
            step = raw_step
        lo     = 0.0 if vmin is None else float(vmin)
        bounds = np.arange(lo, vmax + step, step)
        if bounds[-1] < vmax:
            bounds = np.append(bounds, bounds[-1] + step)
        bounds = np.round(bounds, 6)
        cmap   = plt.get_cmap("jet", len(bounds) - 1)
    else:
        lo     = -vmax if vmin is None else float(vmin)
        hi     = +vmax
        bounds = np.linspace(lo, hi, n_levels + 1)
        base_colors = [
            "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#FFFFFF",
            "#FDDBC7", "#F4A582", "#D6604D", "#B2182B",
        ]
        n_bins = len(bounds) - 1
        colors = base_colors[:n_bins] if n_bins <= len(base_colors) else \
            [ListedColormap(base_colors)(t) for t in np.linspace(0, 1, n_bins)]
        cmap = ListedColormap(colors, name="dynamic_error")
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm, bounds


# =============================================================================
# EMBEDDING FRAMEWORK
# Must match training exactly, including qdm_lr_local / qdm_hr_local [R3]
# =============================================================================
def _cosine_betas(n_steps, device):
    t      = torch.linspace(0, 1, n_steps + 1, device=device)
    alphas = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas = alphas / alphas[0]
    betas  = (1 - alphas[1:] / alphas[:-1]).clamp(max=0.999)
    return betas


def _apply_noise_steps(field, betas, n_steps):
    x = field.clone()
    for t in range(min(n_steps, len(betas))):
        noise = torch.randn_like(x)
        x = (1.0 - betas[t]).sqrt() * x + betas[t].sqrt() * noise
    return x


class EmbeddingFramework:
    """
    Matches the training EmbeddingFramework exactly.
    [R3] load_state_dict now handles qdm_lr_local/qdm_hr_local safely.
    """
    def __init__(self, device, s_freq=0.25, n_noise_steps=10, scale_factor=4):
        self.device        = device
        self.s_freq        = s_freq
        self.n_noise_steps = n_noise_steps
        self.scale_factor  = scale_factor
        self.qdm_lr        = [None] * N_CH
        self.qdm_hr        = [None] * N_CH
        self.qdm_lr_local  = [None] * N_CH   # [R3] present in training state_dict
        self.qdm_hr_local  = [None] * N_CH   # [R3]
        self._fitted       = False
        self.betas         = _cosine_betas(n_noise_steps, device)

    def load_state_dict(self, d):
        self.s_freq        = d["s_freq"]
        self.n_noise_steps = d["n_noise_steps"]
        self.scale_factor  = d["scale_factor"]
        self.qdm_lr        = d["qdm_lr"]
        self.qdm_hr        = d["qdm_hr"]
        # [R3] safe get — old checkpoints may not have local keys
        self.qdm_lr_local  = d.get("qdm_lr_local", [None] * N_CH)
        self.qdm_hr_local  = d.get("qdm_hr_local", [None] * N_CH)
        self._fitted       = d["fitted"]
        self.betas         = _cosine_betas(self.n_noise_steps, self.device)

    def _determine_noise_steps(self):
        return max(10, min(int(self.s_freq * 2.0 * self.n_noise_steps),
                           self.n_noise_steps))

    def _interp(self, arr, q_lr, q_hr):
        return np.interp(arr, q_lr, q_hr)

    def g(self, lr_field, hr_size, weight_map=None):
        """
        Inference-time conditioning — identical to training g().
        weight_map: [1, 1, H_hr, W_hr] Singapore weight map (optional but recommended).
        """
        if not self._fitted:
            raise RuntimeError("EmbeddingFramework not fitted.")
        B, C       = lr_field.shape[:2]
        H_hr, W_hr = hr_size
        lr_up      = F.interpolate(lr_field, size=(H_hr, W_hr),
                                   mode="bilinear", align_corners=False)
        lr_qdm     = torch.empty_like(lr_up)

        for ch in range(C):
            arr         = lr_up[:, ch].cpu().numpy()
            global_map  = self._interp(arr, self.qdm_lr[ch], self.qdm_hr[ch])

            if self.qdm_lr_local[ch] is not None:
                local_map = self._interp(arr, self.qdm_lr_local[ch], self.qdm_hr_local[ch])
            else:
                local_map = global_map

            global_map = torch.tensor(global_map, dtype=lr_field.dtype, device=self.device)
            local_map  = torch.tensor(local_map,  dtype=lr_field.dtype, device=self.device)

            if weight_map is not None:
                w       = weight_map.squeeze(1)
                w       = w / (w.mean() + 1e-8)
                w       = w.clamp(0.2, 8.0)
                w_power = w ** 2.5
                w_power = w_power / (w_power + 1.0)
                lr_qdm[:, ch] = (1.0 - w_power) * global_map + w_power * local_map
            else:
                lr_qdm[:, ch] = global_map

        return _apply_noise_steps(lr_qdm, self.betas, self._determine_noise_steps())


# =============================================================================
# SINGAPORE WEIGHT MAP  (must match training exactly)
# =============================================================================
def build_singapore_weight_map(H, W, lat, lon, device):
    lat_t    = torch.tensor(lat, device=device, dtype=torch.float32)
    lon_t    = torch.tensor(lon, device=device, dtype=torch.float32)
    lat_grid = lat_t[:, None].expand(-1, W)
    lon_grid = lon_t[None, :].expand(H, -1)
    dist     = torch.sqrt(
        (lat_grid - SG_LAT_C) ** 2 +
        ((lon_grid - SG_LON_C) * torch.cos(torch.deg2rad(lat_grid))) ** 2
    )
    weight = 1.0 + (SG_WEIGHT - 1.0) * torch.exp(-(dist / SG_SIGMA) ** 2)
    return weight.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]


# =============================================================================
# EDM / MODEL  (unchanged from training)
# =============================================================================
def get_sigma_schedule(n_steps, device, sigma_min, sigma_max, rho):
    steps  = torch.arange(n_steps + 1, device=device)
    t      = steps / n_steps
    sigmas = (sigma_max**(1/rho) + t*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    return sigmas.clamp(min=0.0)


def edm_precond(sigma, sigma_data):
    sigma   = sigma.reshape(-1, 1, 1, 1)
    c_skip  = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out   = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in    = 1.0 / (sigma_data**2 + sigma**2).sqrt()
    c_noise = sigma.flatten().log() / 4.0
    return c_skip, c_out, c_in, c_noise


class SingaporeCorrDiffNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = DiffusionUNet(in_channels=12, out_channels=4)

    def forward(self, x, t):
        H_in, W_in = x.shape[-2], x.shape[-1]
        out = self.net(x, t)
        if out.shape[-2:] != (H_in, W_in):
            out = F.interpolate(out, size=(H_in, W_in),
                                mode="bilinear", align_corners=False)
        return out


class SingaporeCorrDiffModel(nn.Module):
    def __init__(self, diffusion_net, regressor,
                 sigma_data, sigma_min, sigma_max, rho=_RHO):
        super().__init__()
        self.diffusion_net = diffusion_net
        self.regressor     = regressor
        self.sigma_data    = sigma_data
        self.sigma_min     = sigma_min
        self.sigma_max     = sigma_max
        self.rho           = rho
        for p in self.regressor.parameters():
            p.requires_grad_(False)

    def get_mu(self, lr):
        with torch.no_grad():
            return self.regressor(lr)

    def _mu_at_size(self, lr, H, W):
        mu = self.get_mu(lr)
        if mu.shape[-2:] != (H, W):
            mu = F.interpolate(mu, size=(H, W),
                               mode="bilinear", align_corners=False)
        return mu

    @torch.no_grad()
    def sample(self, lr, n_steps=50, embedding_fw=None, weight_map=None):
        H_hr = lr.shape[-2] * 4
        W_hr = lr.shape[-1] * 4
        mu   = self._mu_at_size(lr, H_hr, W_hr)

        # [R1] Use embedding with weight_map exactly as training does
        if embedding_fw is not None:
            cond_ch = embedding_fw.g(lr, hr_size=(H_hr, W_hr), weight_map=weight_map)
        else:
            cond_ch = F.interpolate(lr, size=(H_hr, W_hr),
                                    mode="bilinear", align_corners=False)

        static_cond = torch.cat([mu, cond_ch], dim=1)
        sigmas      = get_sigma_schedule(n_steps, lr.device,
                                         self.sigma_min, self.sigma_max, self.rho)
        # Match training sample(): x init uses * 0.8
        x = torch.randn_like(mu) * sigmas[0] * 0.8

        def denoise(x_in, sig):
            sig_b          = sig.expand(lr.size(0))
            inp            = torch.cat([x_in, static_cond], dim=1)
            cs, co, ci, cn = edm_precond(sig_b, self.sigma_data)
            Fx             = self.diffusion_net(ci * inp, cn)
            return cs * x_in + co * Fx

        for i in range(n_steps):
            sc  = sigmas[i].clamp(min=0.02)    # match training clamp value
            sn  = sigmas[i + 1].clamp(min=0.02)
            Dx  = denoise(x, sc)
            d   = (x - Dx) / sc
            if i == n_steps - 1:
                x = Dx
            else:
                xn  = x + (sn - sc) * d
                Dx2 = denoise(xn, sn)
                d2  = (xn - Dx2) / sn
                x   = x + (sn - sc) * (d + d2) * 0.5

        # Match training sample() output: clamp residual then add mu, clamp >= 0
        return (mu + x.clamp(-1.4, 1.8)).clamp(min=0.0)


# =============================================================================
# MODEL LOADER  -- [R1] + [R2] fixes here
# =============================================================================
def _torch_load(path, device):
    import numpy as np
    import numpy._core.multiarray
    _NP_SAFE = [np.ndarray, np.dtype,
                numpy._core.multiarray._reconstruct,
                numpy._core.multiarray.scalar]
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        pass
    try:
        with torch.serialization.safe_globals(_NP_SAFE):
            return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        pass
    print(f"  Warning: loading {path} with weights_only=False")
    return torch.load(path, map_location=device, weights_only=False)


def load_model(device):
    ckpt = _torch_load(BEST_CKPT, device)

    sigma_data = ckpt.get("sigma_data", 0.61)
    sigma_min  = ckpt.get("sigma_min",  sigma_data * 0.01)
    sigma_max  = ckpt.get("sigma_max",  sigma_data * 6.0)

    reg = ClimateRegressor().to(device)
    reg.load_state_dict(_torch_load(REG_CKPT, device))
    reg.eval()

    dnet  = SingaporeCorrDiffNet().to(device)
    model = SingaporeCorrDiffModel(dnet, reg, sigma_data, sigma_min, sigma_max).to(device)

    raw   = (ckpt.get("model_state_dict") or ckpt.get("model") or
             ckpt.get("state_dict") or ckpt)
    clean = {(k[7:] if k.startswith("module.") else k): v for k, v in raw.items()}
    missing, unexpected = model.diffusion_net.load_state_dict(clean, strict=False)
    print(f"Model loaded  |  missing={len(missing)}  unexpected={len(unexpected)}")
    print(f"  sigma_data={sigma_data:.4f}  sigma_min={sigma_min:.4f}  "
          f"sigma_max={sigma_max:.4f}")

    # ------------------------------------------------------------------
    # [R1] Training saves embedding under key "embedding", NOT "embedding_state"
    # [R2] No "use_embedding" flag in training checkpoint; just check presence
    # ------------------------------------------------------------------
    embedding_fw = None
    emb_state = (ckpt.get("embedding") or        # <-- training key
                 ckpt.get("embedding_state") or   # <-- old eval key (fallback)
                 None)

    if emb_state is not None and emb_state.get("fitted", False):
        embedding_fw = EmbeddingFramework(device)
        embedding_fw.load_state_dict(emb_state)
        print(f"  EmbeddingFramework loaded: s_freq={embedding_fw.s_freq:.4f}  "
              f"n_noise_steps={embedding_fw.n_noise_steps}  "
              f"noise_steps_used={embedding_fw._determine_noise_steps()}")
        # Verify QDM arrays are present
        if embedding_fw.qdm_lr[PRECIP_CH] is None:
            print("  WARNING: QDM arrays are None — embedding will not apply correction!")
    else:
        print("  WARNING: No valid embedding found in checkpoint!")
        print("  This means conditioning will differ from training -> bad predictions.")
        print(f"  Checkpoint keys: {list(ckpt.keys())}")

    model.eval()
    return model, embedding_fw


# =============================================================================
# GEO
# =============================================================================
def load_geo():
    try:
        ds = xr.open_dataset(LR_FILES[3])
        lat_name = next((c for c in ["lat","latitude","LAT","LATITUDE","y"]
                         if c in ds.coords or c in ds.dims), None)
        lon_name = next((c for c in ["lon","longitude","LON","LONGITUDE","x"]
                         if c in ds.coords or c in ds.dims), None)
        lat = ds[lat_name].values.astype(float) if lat_name else None
        lon = ds[lon_name].values.astype(float) if lon_name else None
        ds.close()
    except Exception:
        lat, lon = None, None

    if lat is None: lat = np.linspace(-2.0, 4.5, 256)
    if lon is None: lon = np.linspace(100.5, 107.0, 276)

    shp = None
    if os.path.exists(SHAPEFILE):
        shp = gpd.read_file(SHAPEFILE)
        shp = (shp.set_crs("EPSG:4326") if shp.crs is None
               else shp.to_crs("EPSG:4326"))
        print("Loaded Singapore boundary shapefile")

    world = None
    if os.path.exists(WORLD_COUNTRIES_SHP):
        try:
            world = gpd.read_file(WORLD_COUNTRIES_SHP)
            world = world.to_crs("EPSG:4326")
            print(f"Loaded {len(world)} countries from Natural Earth")
        except Exception as e:
            print(f"Warning: Could not load world countries: {e}")

    full_ext = [float(lon.min()), float(lon.max()),
                float(lat.min()), float(lat.max())]
    clip_ext = [SG_LON_MIN, SG_LON_MAX, SG_LAT_MIN, SG_LAT_MAX]
    return lat, lon, full_ext, clip_ext, shp, world


def clip_field(field, lat, lon, extent):
    """Clip 2D field. Requires lat to be ASCENDING (call orient_field first)."""
    lon_min, lon_max, lat_min, lat_max = extent
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return field, lat, lon
    r0, r1 = lat_idx[0], lat_idx[-1] + 1
    c0, c1 = lon_idx[0], lon_idx[-1] + 1
    return field[r0:r1, c0:c1], lat[r0:r1], lon[c0:c1]


def orient_field(field_2d, lat_arr):
    """[F3] Ensure field rows are south-to-north (ascending lat)."""
    if lat_arr[0] > lat_arr[-1]:
        return field_2d[::-1, :].copy(), lat_arr[::-1].copy()
    return field_2d, lat_arr


def make_latlon(hr_shape, lat_lr, lon_lr):
    """[F1] Build HR lat/lon grids from ACTUAL model output shape."""
    H, W    = hr_shape
    lat_min = float(min(lat_lr.min(), lat_lr.max()))
    lat_max = float(max(lat_lr.min(), lat_lr.max()))
    lon_min = float(min(lon_lr.min(), lon_lr.max()))
    lon_max = float(max(lon_lr.min(), lon_lr.max()))
    lat_hr  = np.linspace(lat_min, lat_max, H)   # always ascending
    lon_hr  = np.linspace(lon_min, lon_max, W)
    return lat_hr, lon_hr


def load_time():
    ds = xr.open_dataset(LR_FILES[3])
    time_name = next((c for c in ["time", "TIME", "date"]
                      if c in ds.coords or c in ds.dims), None)
    if time_name is None:
        raise RuntimeError("No time coordinate found")
    time = ds[time_name].values
    ds.close()
    return time


def idx_to_date(idx, time_array):
    return str(pd.to_datetime(time_array[idx]).date())


def to_mm(arr_norm, mean, std):
    """[F2] Invert log1p normalisation -> mm/day. Clips negative values."""
    return np.expm1(arr_norm * std + mean).clip(0)


# =============================================================================
# ANALYTICS
# =============================================================================
def compute_pcc(a, b):
    a, b = np.asarray(a).flatten(), np.asarray(b).flatten()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float((pearsonr(a, b)[0] + 1.0) / 2.0)


# =============================================================================
# STYLE + PLOT HELPERS
# =============================================================================
def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "black",   "axes.labelcolor": "black",
        "xtick.color": "black",      "ytick.color": "black",
        "text.color": "black",       "grid.color": "gray",
        "grid.linewidth": 0.5,       "legend.frameon": False,
        "font.family": "serif",      "font.size": 24,
        "axes.titlesize": 22,        "axes.labelsize": 20,
    })


def _pixel_grid(ax, field_shape, extent, color="white"):
    H, W = field_shape
    lon_min, lon_max, lat_min, lat_max = extent
    dx = (lon_max - lon_min) / W
    dy = (lat_max - lat_min) / H
    kw = dict(color=color, linewidth=0.2, alpha=0.3, zorder=5)
    for i in range(0, W + 1):
        ax.axvline(lon_min + i * dx, **kw)
    for j in range(0, H + 1):
        ax.axhline(lat_min + j * dy, **kw)


def _save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {path}")


def build_map_fig(gt, s2, extent, shp, world_gdf=None, title_tag=""):
    rain_vmax = float(np.nanpercentile(np.concatenate([gt.ravel(), s2.ravel()]), 99.5))
    rain_vmax = max(rain_vmax, 1.0)
    cmap_r, norm_r, _ = make_dynamic_cmap(rain_vmax, n_levels=10, kind="rain")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
    for i, (ax, data, title, res) in enumerate(zip(
            axes, [gt, s2],
            ["Ground Truth (8km)", "S2 Diffusion (2km)"],
            ["8km", "2km"])):
        im = ax.imshow(data, origin="lower", extent=extent,
                       cmap=cmap_r, norm=norm_r, interpolation="bilinear")
        if world_gdf is not None:
            world_gdf.boundary.plot(ax=ax, color="black", linewidth=0.8,
                                    alpha=0.7, zorder=10)
        if shp is not None:
            shp.boundary.plot(ax=ax, color="red", linewidth=1.2, zorder=11)
        _pixel_grid(ax, data.shape, extent)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(title, fontsize=20, weight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=12)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("mm/day", fontsize=12)
    fig.suptitle(f"Domain Comparison: {title_tag}", fontsize=22, weight="bold")
    plt.tight_layout()
    return fig


def build_sg_map_fig(gt, pred, extent, shp, world_gdf=None, title_tag=""):
    rain_vmax = float(np.nanpercentile(np.concatenate([gt.ravel(), pred.ravel()]), 99))
    rain_vmax = max(rain_vmax, 1.0)
    cmap_r, norm_r, _ = make_dynamic_cmap(rain_vmax, kind="rain")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
    for i, (ax, data, title, res) in enumerate(zip(
            axes, [gt, pred],
            ["Ground Truth (8km)", "S2 Diffusion (2km)"],
            ["8km", "2km"])):
        ax.set_facecolor("white")
        im = ax.imshow(data, origin="lower", extent=extent,
                       cmap=cmap_r, norm=norm_r, interpolation="bilinear")
        if world_gdf is not None:
            world_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0,
                                    alpha=0.7, zorder=10)
        if shp is not None:
            shp.boundary.plot(ax=ax, color="red", linewidth=1.5, zorder=11)
        _pixel_grid(ax, data.shape, extent)
        ax.set_title(title, color="black", fontsize=22, weight="bold", pad=10)
        ax.set_xlabel("Longitude (°E)", fontsize=14)
        if i == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=14)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=12)
        cb.set_label("Precipitation (mm/day)", fontsize=12)

    pcc = compute_pcc(pred, gt)
    fig.suptitle(f"{title_tag} | Singapore Focus (PCC: {pcc:.3f})",
                 fontsize=24, color="black", weight="bold", y=0.98)
    fig.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================
def main():
    apply_style()
    time_array = load_time()

    print("Loading dataset...")
    dataset = ClimateDataset(LR_FILES, LR_FILES, rank=0)

    PRECIP_MEAN = dataset.mean["precip"]
    PRECIP_STD  = dataset.std["precip"]
    print(f"  PRECIP_MEAN={PRECIP_MEAN:.4f}  PRECIP_STD={PRECIP_STD:.4f}")

    # -- Data split (must match training exactly) ----------------------------
    n = len(dataset)
    gen = torch.Generator().manual_seed(42)
    _, _, test_set = random_split(
        dataset,
        [int(0.65*n), int(0.15*n), n - int(0.65*n) - int(0.15*n)],
        generator=gen,
    )
    test_indices = test_set.indices

    # -- Geo -----------------------------------------------------------------
    print("Loading geo...")
    lat, lon, full_ext, clip_ext, shp, world_gdf = load_geo()

    # [F3] Determine orientation once from raw lat
    lat_is_descending = bool(lat[0] > lat[-1])
    print(f"  lat[0]={lat[0]:.3f}  lat[-1]={lat[-1]:.3f}  "
          f"descending={lat_is_descending}")

    WET_THRESH_MM = 0.01

    # -- Model ---------------------------------------------------------------
    print("Loading model...")
    model, embedding_fw = load_model(DEVICE)

    if embedding_fw is None:
        print("\n" + "="*60)
        print("CRITICAL: embedding_fw is None.")
        print("Model was trained with QDM embedding conditioning.")
        print("Predictions will likely be near-zero without it.")
        print("Check checkpoint keys above.")
        print("="*60 + "\n")

    # -- Build Singapore weight map (for sample() conditioning) --------------
    # We need it at HR size. Use a dummy pass to get H_hr, W_hr.
    _, hr_sample = dataset[test_indices[0]]
    H_hr, W_hr = hr_sample.shape[-2], hr_sample.shape[-1]
    del hr_sample

    lat_hr_full, lon_hr_full = make_latlon((H_hr, W_hr), lat, lon)
    singapore_wmap = build_singapore_weight_map(
        H_hr, W_hr, lat_hr_full, lon_hr_full, DEVICE
    )
    print(f"  Singapore weight map: {H_hr}x{W_hr}  "
          f"peak={singapore_wmap.max().item():.2f}x")

    # -- Find top rainfall test samples over Singapore -----------------------
    print("Searching for top rainfall test samples...")
    sg_candidates = []

    for i in range(len(test_set)):
        dataset_idx = test_indices[i]
        _, hr_s = dataset[dataset_idx]

        rain_mm = to_mm(hr_s[PRECIP_CH].numpy(), PRECIP_MEAN, PRECIP_STD)

        # [F3] orient before clipping
        if lat_is_descending:
            rain_mm = rain_mm[::-1, :].copy()

        lat_hr, lon_hr = make_latlon(rain_mm.shape, lat, lon)
        rain_sg, _, _  = clip_field(rain_mm, lat_hr, lon_hr, clip_ext)

        if rain_sg.size > 0:
            v = float(np.nanmax(rain_sg))
            if v > WET_THRESH_MM:
                sg_candidates.append((v, dataset_idx))

    sg_candidates.sort(reverse=True)
    top_samples = sg_candidates[:10]
    print(f"  Found {len(sg_candidates)} wet samples; using top {len(top_samples)}")

    # -- Generate maps -------------------------------------------------------
    for rank, (best_val, best_idx) in enumerate(top_samples):
        date_str = idx_to_date(best_idx, time_array)
        tag      = f"rank{rank+1}_{date_str}"
        print(f"\n--- #{rank+1}  {date_str}  Max SG Rain: {best_val:.2f} mm ---")

        lr_s, hr_s = dataset[best_idx]
        lr_t = lr_s.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # [R1] Pass weight_map so embedding conditioning matches training
            s2 = model.sample(lr_t, n_steps=N_INFER,
                              embedding_fw=embedding_fw,
                              weight_map=singapore_wmap)

        gt_norm = hr_s[PRECIP_CH].numpy()
        s2_norm = s2[0, PRECIP_CH].cpu().numpy()

        print(f"  Normalized  GT:   [{gt_norm.min():.3f}, {gt_norm.max():.3f}]")
        print(f"  Normalized  Pred: [{s2_norm.min():.3f}, {s2_norm.max():.3f}]")

        gt_mm = to_mm(gt_norm, PRECIP_MEAN, PRECIP_STD)
        s2_mm = to_mm(s2_norm, PRECIP_MEAN, PRECIP_STD)

        print(f"  mm/day      GT:   [{gt_mm.min():.2f}, {gt_mm.max():.2f}]")
        print(f"  mm/day      Pred: [{s2_mm.min():.2f}, {s2_mm.max():.2f}]")

        # [F1] HR lat/lon from actual shapes
        lat_hr_gt, lon_hr_gt   = make_latlon(gt_mm.shape, lat, lon)
        lat_hr_s2, lon_hr_s2   = make_latlon(s2_mm.shape, lat, lon)

        # [F3] Orient consistently
        if lat_is_descending:
            gt_mm = gt_mm[::-1, :].copy()
            s2_mm = s2_mm[::-1, :].copy()

        # Full domain figure
        fig_full = build_map_fig(gt_mm, s2_mm, full_ext, shp, world_gdf,
                                 title_tag=tag)
        _save(fig_full, os.path.join(OUTPUT_DIR, f"map_full_{tag}.png"))

        # Singapore zoomed figure
        gt_sg, _, _ = clip_field(gt_mm, lat_hr_gt, lon_hr_gt, clip_ext)
        s2_sg, _, _ = clip_field(s2_mm, lat_hr_s2, lon_hr_s2, clip_ext)

        print(f"  Singapore   GT:   [{gt_sg.min():.2f}, {gt_sg.max():.2f}]")
        print(f"  Singapore   Pred: [{s2_sg.min():.2f}, {s2_sg.max():.2f}]")

        if gt_sg.size == 0 or s2_sg.size == 0:
            print("  WARNING: empty Singapore clip — check lat/lon bounds!")
            continue

        fig_sg = build_sg_map_fig(gt_sg, s2_sg, clip_ext, shp, world_gdf,
                                  title_tag=tag)
        _save(fig_sg, os.path.join(OUTPUT_DIR, f"map_sg_{tag}.png"))

    print("\nAll plots generated in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
