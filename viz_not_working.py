# -*- coding: utf-8 -*-
# EvalCorrDiff_Singapore_v4.py  (fixed v2)
#
# Changes vs previous fix:
#   [C1]  PSD: second curve is now coarse LR upsampled input (not regressor mu).
#   [C2]  PCC figures: pure gaussian_kde only (no histogram fallback).
#   [C3]  All error panels: pure signed error (pred - gt), symmetric diverging
#         cmap; no abs() anywhere in the error display path.
#   [C4]  main: regressor / s1 evaluation paths removed; lr_up_mm collected
#         for PSD; Singapore map is the primary visual focus.

import os
import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
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
             
             
# Add this right after SHAPEFILE definition
WORLD_COUNTRIES_SHP = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

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

TEXT  = "black"
BG    = "white"
GRID  = "gray"
SUB   = "black"
PANEL = "white"


# =============================================================================
# [BF1] COLOURMAP  -- fixed 'error' branch (was referencing undefined vars)
# =============================================================================
def make_dynamic_cmap(vmax, n_levels=9, kind="rain",
                      vmin=None):
    """
    Dynamic discretized colormap with robust bounds.

    Parameters
    ----------
    vmax      : upper bound (also used as +/- symmetric limit for 'error')
    n_levels  : number of discrete colour bins
    kind      : 'rain'  -> jet colormap, bounds [0, vmax]
                'error' -> diverging white-to-red, bounds [-vmax, +vmax]
                           (symmetric; pass vmin to override lower bound)
    vmin      : explicit lower bound override (default: 0 for rain, -vmax for error)

    Returns
    -------
    cmap, norm, bounds
    """
    vmax = max(float(vmax), 1e-6)

    if kind == "rain":
        # --- precipitation: [0, vmax] with nice step ---
        raw_step  = vmax / max(n_levels - 1, 1)
        magnitude = 10 ** math.floor(math.log10(max(raw_step, 1e-9)))
        step      = magnitude * round(raw_step / magnitude)
        if step <= 0:
            step = raw_step

        lo = 0.0 if vmin is None else float(vmin)
        bounds = np.arange(lo, vmax + step, step)
        if bounds[-1] < vmax:
            bounds = np.append(bounds, bounds[-1] + step)
        bounds = np.round(bounds, 6)

        cmap = plt.get_cmap("jet", len(bounds) - 1)

    else:
        # --- error / bias: symmetric [-vmax, +vmax] ---
        lo = -vmax if vmin is None else float(vmin)
        hi = +vmax
        bounds = np.linspace(lo, hi, n_levels + 1)   # [BF1] was using undefined err_vmin/err_vmax

        base_colors = [
            "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0",
            "#FFFFFF",
            "#FDDBC7", "#F4A582", "#D6604D", "#B2182B",
        ]
        n_bins = len(bounds) - 1
        if n_bins <= len(base_colors):
            colors = base_colors[:n_bins]
        else:
            cmap_base = ListedColormap(base_colors)
            colors = [cmap_base(t) for t in np.linspace(0, 1, n_bins)]

        cmap = ListedColormap(colors, name="dynamic_error")

    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm, bounds


# =============================================================================
# EMBEDDING FRAMEWORK  (unchanged)
# =============================================================================
def create_sg_mask(lat, lon, shp):
    H, W = len(lat), len(lon)
    mask = np.zeros((H, W), dtype=bool)
    geom = shp.unary_union
    for i in range(H):
        for j in range(W):
            if geom.contains(Point(lon[j], lat[i])):
                mask[i, j] = True
    return mask


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
    def __init__(self, device):
        self.device        = device
        self.s_freq        = 0.25
        self.n_noise_steps = 50
        self.scale_factor  = 4
        self.qdm_lr        = [None] * N_CH
        self.qdm_hr        = [None] * N_CH
        self._fitted       = False
        self.betas         = _cosine_betas(self.n_noise_steps, device)

    def load_state_dict(self, d):
        self.s_freq        = d["s_freq"]
        self.n_noise_steps = d["n_noise_steps"]
        self.scale_factor  = d["scale_factor"]
        self.qdm_lr        = d["qdm_lr"]
        self.qdm_hr        = d["qdm_hr"]
        self._fitted       = d["fitted"]
        self.betas         = _cosine_betas(self.n_noise_steps, self.device)

    def _determine_noise_steps(self):
        steps = int(self.s_freq * 2.0 * self.n_noise_steps)
        return max(5, min(steps, self.n_noise_steps))

    def _apply_qdm_channel(self, field_ch, ch):
        arr    = field_ch.cpu().numpy()
        mapped = np.interp(arr, self.qdm_lr[ch], self.qdm_hr[ch])
        return torch.tensor(mapped, dtype=field_ch.dtype, device=self.device)

    def g(self, lr_field, hr_size):
        if not self._fitted:
            raise RuntimeError("EmbeddingFramework not loaded from checkpoint.")
        B, C       = lr_field.shape[:2]
        H_hr, W_hr = hr_size
        lr_qdm = torch.empty(B, C, *lr_field.shape[2:],
                             dtype=lr_field.dtype, device=self.device)
        for ch in range(C):
            lr_qdm[:, ch] = self._apply_qdm_channel(lr_field[:, ch], ch)
        lr_up   = F.interpolate(lr_qdm, size=(H_hr, W_hr),
                                mode="bilinear", align_corners=False)
        n_steps = self._determine_noise_steps()
        return _apply_noise_steps(lr_up, self.betas, n_steps)


# =============================================================================
# MODEL  (unchanged)
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
    def sample(self, lr, n_steps=50, embedding_fw=None):
        H_hr = lr.shape[-2] * 4
        W_hr = lr.shape[-1] * 4
        mu   = self._mu_at_size(lr, H_hr, W_hr)

        if embedding_fw is not None:
            cond_ch = embedding_fw.g(lr, hr_size=(H_hr, W_hr))
        else:
            cond_ch = F.interpolate(lr, size=(H_hr, W_hr),
                                    mode="bilinear", align_corners=False)

        static_cond = torch.cat([mu, cond_ch], dim=1)
        sigmas      = get_sigma_schedule(n_steps, lr.device,
                                         self.sigma_min, self.sigma_max, self.rho)
        x = torch.randn_like(mu) * sigmas[0].clamp(min=1e-5)

        def denoise(x_in, sig):
            sig_b          = sig.expand(lr.size(0))
            inp            = torch.cat([x_in, static_cond], dim=1)
            cs, co, ci, cn = edm_precond(sig_b, self.sigma_data)
            Fx             = self.diffusion_net(ci * inp, cn)
            return cs * x_in + co * Fx

        for i in range(n_steps):
            sc  = sigmas[i].clamp(min=1e-5)
            sn  = sigmas[i + 1].clamp(min=1e-5)
            Dx  = denoise(x, sc)
            d   = (x - Dx) / sc
            if i == n_steps - 1:
                x = Dx
            else:
                xn  = x + (sn - sc) * d
                Dx2 = denoise(xn, sn)
                d2  = (xn - Dx2) / sn
                x   = x + (sn - sc) * (d + d2) * 0.5

        return mu + x


def _torch_load(path, device):
    import numpy as np
    import numpy._core.multiarray

    _NP_SAFE_GLOBALS = [
        np.ndarray,
        np.dtype,
        numpy._core.multiarray._reconstruct,
        numpy._core.multiarray.scalar,
    ]

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        pass

    try:
        with torch.serialization.safe_globals(_NP_SAFE_GLOBALS):
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

    raw = (ckpt.get("model_state_dict") or ckpt.get("model") or
           ckpt.get("state_dict") or ckpt)
    clean = {(k[7:] if k.startswith("module.") else k): v for k, v in raw.items()}
    missing, unexpected = model.diffusion_net.load_state_dict(clean, strict=False)
    print(f"Model loaded  |  missing={len(missing)}  unexpected={len(unexpected)}")
    print(f"  sigma_data={sigma_data:.4f}  sigma_min={sigma_min:.4f}  sigma_max={sigma_max:.4f}")

    embedding_fw = None
    emb_state    = ckpt.get("embedding_state", None)
    use_emb      = ckpt.get("use_embedding", False)

    if use_emb and emb_state is not None and emb_state.get("fitted", False):
        embedding_fw = EmbeddingFramework(device)
        embedding_fw.load_state_dict(emb_state)
        print(f"  EmbeddingFramework loaded: s_freq={embedding_fw.s_freq:.4f}  "
              f"noise_steps={embedding_fw._determine_noise_steps()}")
    else:
        print("  No embedding_state in checkpoint -- using raw lr_up (v2 mode).")

    model.eval()
    return model, embedding_fw


# =============================================================================
# GEO  (unchanged)
# =============================================================================

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

    # Singapore boundary (your original)
    shp = None
    if os.path.exists(SHAPEFILE):
        shp = gpd.read_file(SHAPEFILE)
        shp = (shp.set_crs("EPSG:4326") if shp.crs is None
               else shp.to_crs("EPSG:4326"))
        print(f"Loaded Singapore boundary shapefile")

    # === NEW: Load all world countries ===
    world = None
    if os.path.exists(WORLD_COUNTRIES_SHP):
        try:
            world = gpd.read_file(WORLD_COUNTRIES_SHP)
            world = world.to_crs("EPSG:4326")
            print(f"✅ Loaded {len(world)} countries from Natural Earth (10m resolution)")
        except Exception as e:
            print(f"Warning: Could not load world countries: {e}")
    else:
        print(f"Warning: World countries shapefile not found at {WORLD_COUNTRIES_SHP}")

    full_ext = [float(lon.min()), float(lon.max()),
                float(lat.min()), float(lat.max())]
    clip_ext = [SG_LON_MIN, SG_LON_MAX, SG_LAT_MIN, SG_LAT_MAX]

    return lat, lon, full_ext, clip_ext, shp, world   # <-- Now returns 6 values

def clip_field(field, lat, lon, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return field, lat, lon
    r0, r1 = lat_idx[0], lat_idx[-1]+1
    c0, c1 = lon_idx[0], lon_idx[-1]+1
    return field[r0:r1, c0:c1], lat[r0:r1], lon[c0:c1]


def expand_extent(extent, expand_deg=1.0):
    lon_min, lon_max, lat_min, lat_max = extent
    return [lon_min - expand_deg, lon_max + expand_deg,
            lat_min - expand_deg, lat_max + expand_deg]


def to_mm(arr_norm, mean, std):
    """Invert log1p normalisation: x_mm = expm1(arr_norm * std + mean)."""
    x_log = arr_norm * std + mean
    return np.expm1(x_log)


def load_time():
    ds = xr.open_dataset(LR_FILES[3])
    time_name = next((c for c in ["time", "TIME", "date"]
                      if c in ds.coords or c in ds.dims), None)
    if time_name is None:
        raise RuntimeError("No time coordinate found in dataset")
    time = ds[time_name].values
    ds.close()
    return time


def idx_to_date(idx, time_array):
    return str(pd.to_datetime(time_array[idx]).date())


def make_latlon(shape, lat, lon):
    """
    Generates high-resolution 2 km coordinates based on the 8 km input bounds.
    """
    H, W = shape # This should be the HR shape (e.g., 256x276)
    
    # Generate linear spacing for the 2 km grid
    lat_hr = np.linspace(float(lat.min()), float(lat.max()), H)
    lon_hr = np.linspace(float(lon.min()), float(lon.max()), W)
    
    # Maintain orientation consistency
    if lat[0] > lat[-1] and lat_hr[0] < lat_hr[-1]:
        lat_hr = lat_hr[::-1]
        
    return lat_hr, lon_hr


# =============================================================================
# INTENSITY METRICS  (unchanged)
# =============================================================================

def intensity_error(pred_mm, gt_mm, wet_thresh_mm):
    wet = gt_mm > wet_thresh_mm
    if wet.sum() == 0:
        return float("nan")
    gt_max   = float(np.max(gt_mm[wet]))
    pred_max = float(np.max(pred_mm[wet]))
    return 100.0 * abs(pred_max - gt_max) / (gt_max + 1e-6)


# =============================================================================
# ANALYTICS  (unchanged)
# =============================================================================

def compute_psd_2d(field):
    field = np.asarray(field)
    if field.ndim == 3:
        field = field[0]
    f   = field - np.mean(field)
    fft = np.fft.fft2(f)
    return np.fft.fftshift((np.abs(fft) ** 2) / f.size)


def radial_average(psd2d):
    h, w = psd2d.shape
    cy, cx = h // 2, w // 2
    y, x   = np.indices((h, w))
    r      = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    tbin   = np.bincount(r.ravel(), psd2d.ravel())
    nr     = np.bincount(r.ravel())
    return (tbin / (nr + 1e-8))[1:min(h, w)//2]


def compute_mean_psd(fields):
    psds    = [radial_average(compute_psd_2d(f)) for f in fields]
    min_len = min(len(p) for p in psds)
    return np.array([p[:min_len] for p in psds]).mean(axis=0)


def compute_pcc(a, b):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float((pearsonr(a, b)[0] + 1.0) / 2.0)


def compute_pcc_sg(a, b, lat, lon, extent):
    a_sg, lat_sg, lon_sg = clip_field(a, lat, lon, extent)
    b_sg, _, _           = clip_field(b, lat, lon, extent)
    if a_sg.size == 0 or b_sg.size == 0:
        return 0.0
    af, bf = a_sg.flatten(), b_sg.flatten()
    if af.std() < 1e-8 or bf.std() < 1e-8:
        return 0.0
    return float((pearsonr(af, bf)[0] + 1.0) / 2.0)


# manual_pdf removed � KDE is used exclusively for PCC figures [C2]


# =============================================================================
# STYLE
# =============================================================================

def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "black",
        "axes.labelcolor":  "black",
        "xtick.color":      "black",
        "ytick.color":      "black",
        "text.color":       "black",
        "grid.color":       "gray",
        "grid.linewidth":   0.5,
        "legend.frameon":   False,
        "font.family":      "serif",
        "font.size":        25,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
    })


def _shp(ax, shp):
    if shp is not None:
        shp.boundary.plot(ax=ax, color="#000000",
                          linewidth=2.0, alpha=1.0, zorder=6)
                          
                          
                          
def _plot_country_borders(ax, world_gdf, color="#333333", linewidth=1.1, alpha=0.85, zorder=5):
    """Plot borders of all countries in the region"""
    if world_gdf is None:
        return
    world_gdf.boundary.plot(ax=ax, color=color, linewidth=linewidth, 
                           alpha=alpha, zorder=zorder)


def _pixel_grid(ax, field_shape, extent, color="gray", grid_type="2km"):
    H, W = field_shape
    lon_min, lon_max, lat_min, lat_max = extent
    
    dx = (lon_max - lon_min) / W
    dy = (lat_max - lat_min) / H
    
    # If it's the 8km panel, the lines will be further apart
    # If it's the 2km panel, they will be 4x denser
    kw = dict(color=color, linewidth=0.2, alpha=0.3, zorder=5)
    
    for i in range(0, W + 1):
        ax.axvline(lon_min + i * dx, **kw)
    for j in range(0, H + 1):
        ax.axhline(lat_min + j * dy, **kw)

def _discrete_cb(fig, ax, im, cmap, norm, bounds, label):
    cb = fig.colorbar(im, ax=ax, cmap=cmap, norm=norm,
                      fraction=0.038, pad=0.02,
                      ticks=bounds, spacing="uniform")
    cb.set_label(label, color=TEXT, fontsize=20)
    tick_labels = [f"{b:.0f}" if b == int(b) else f"{b:.1f}" for b in bounds]
    cb.ax.set_yticklabels(tick_labels, color=SUB, fontsize=18)
    cb.outline.set_edgecolor(GRID)
    return cb


def _save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {path}")

# =============================================================================
# STYLE - Updated for clean white look
# =============================================================================
def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "grid.color": "gray",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "font.family": "serif",
        "font.size": 24,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
    })
def build_map_fig(gt, s2, extent, shp, world_gdf=None, title_tag="", grid_spacing=1.0):
    """
    Shows Ground Truth (8km) vs S2 Diffusion (2km) constrained to the domain.
    """
    # 1. Robust Vmax calculation
    rain_vmax = float(np.nanpercentile(np.concatenate([gt.ravel(), s2.ravel()]), 99.5))
    rain_vmax = max(rain_vmax, 1.0)
    
    cmap_r, norm_r, bounds_r = make_dynamic_cmap(rain_vmax, n_levels=10, kind="rain")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
    
    data = [gt, s2]
    titles = ["Ground Truth (8km)", "S2 Diffusion (2km)"]
    resolutions = ["8km", "2km"]

    for i, ax in enumerate(axes):
        # Use origin="lower" to match standard geospatial plotting
        im = ax.imshow(data[i], origin="lower", extent=extent,
                       cmap=cmap_r, norm=norm_r, interpolation="bilinear")

        # Plot country borders
        if world_gdf is not None:
            world_gdf.boundary.plot(ax=ax, color="black", linewidth=0.8, alpha=0.7, zorder=10)
        
        # Plot Singapore highlight
        if shp is not None:
            shp.boundary.plot(ax=ax, color="red", linewidth=1.2, zorder=11)

        # APPLY PIXEL GRID
        _pixel_grid(ax, data[i].shape, extent, color="white", grid_type=resolutions[i])

        # --- CRITICAL: ZOOM TO DOMAIN ---
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        ax.set_title(titles[i], fontsize=20, weight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=12)
        
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("mm/day", fontsize=12)

    fig.suptitle(f"Domain Comparison: {title_tag}", fontsize=22, weight="bold")
    plt.tight_layout()
    return fig

def build_sg_map_fig(gt, pred, extent, shp, world_gdf=None, title_tag=""):
    """
    Singapore-focused map - Shows Ground Truth (8km) vs Model (2km).
    Removes the error/bias panel and adds resolution-specific pixel grids.
    """
    # Calculate robust max for precipitation colorbar
    rain_vmax = float(np.nanpercentile(np.concatenate([gt.ravel(), pred.ravel()]), 99))
    rain_vmax = max(rain_vmax, 1.0)

    cmap_r, norm_r, bounds_r = make_dynamic_cmap(rain_vmax, kind="rain")

    # Change to 1 row, 2 columns for a cleaner comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
    fig.patch.set_facecolor("white")

    # Define titles and resolutions for the grid logic
    titles = ["Ground Truth (8km)", "S2 Diffusion (2km)"]
    data = [gt, pred]
    res_tags = ["8km", "2km"]

    for i, ax in enumerate(axes):
        ax.set_facecolor("white")
        
        # Display the rainfall data
        im = ax.imshow(data[i], origin="lower", extent=extent,
                       cmap=cmap_r, norm=norm_r, interpolation="bilinear")

        # 1. Plot Regional Country Borders
        if world_gdf is not None:
            world_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0, 
                                    alpha=0.7, zorder=10)
        
        # 2. Highlight Singapore Boundary (Red for focus)
        if shp is not None:
            shp.boundary.plot(ax=ax, color="red", linewidth=1.5, zorder=11)

        # 3. APPLY PIXEL GRID: 8km for GT, 2km for Model
        # This uses the shape of the data to draw the grid lines
        _pixel_grid(ax, data[i].shape, extent, color="white", grid_type=res_tags[i])

        # Formatting
        ax.set_title(titles[i], color="black", fontsize=22, weight="bold", pad=10)
        ax.set_xlabel("Longitude (°E)", fontsize=14)
        if i == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=14)
        
        # Set limits to the requested Singapore extent
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Add colorbar to each panel
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=12)
        cb.set_label("Precipitation (mm/day)", fontsize=12)

    # Global Stats (Optional: moved to subtitle or removed)
    pcc = compute_pcc(pred, gt)
    fig.suptitle(f"{title_tag} | Singapore Focus (PCC: {pcc:.3f})", 
                 fontsize=24, color="black", weight="bold", y=0.98)
    
    fig.tight_layout()
    return fig
# =============================================================================
# [C4] MAIN  -- regressor / s1 paths removed; lr_up_mm for PSD; SG map focus
# =============================================================================
def main():
    apply_style()
    time_array = load_time()

    print("Loading dataset...")
    dataset = ClimateDataset(LR_FILES, LR_FILES, rank=0)

    global PRECIP_MEAN, PRECIP_STD
    PRECIP_MEAN = dataset.mean["precip"]
    PRECIP_STD  = dataset.std["precip"]

    # -- Data split ------------------------------------------------------------
    n = len(dataset)
    gen = torch.Generator().manual_seed(42)
    _, _, test_set = random_split(
        dataset,
        [int(0.65*n), int(0.15*n), n - int(0.65*n) - int(0.15*n)],
        generator=gen,
    )
    test_indices = test_set.indices
    test_indices_set = set(test_indices)

    # -- Geo -------------------------------------------------------------------
    print("Loading geo...")
    lat, lon, full_ext, clip_ext, shp, world_gdf = load_geo()

    # Create the high-res coordinate arrays once
    lat_hr_full, lon_hr_full = make_latlon((256, 276), lat, lon)
    
    # Check if latitudes are descending (common in some NetCDF files)
    # We want them ascending for origin="lower"
    if lat_hr_full[0] > lat_hr_full[-1]:
        lat_hr_full = lat_hr_full[::-1]

    WET_THRESH_MM = 0.01

    # -- Model -----------------------------------------------------------------
    print("Loading model...")
    model, embedding_fw = load_model(DEVICE)

    print("Searching for top rainfall samples over Singapore...")
    sg_candidates = []

    # Optimization: Only check a subset or use pre-calculated indices if possible
    for i in range(len(test_set)):
        dataset_idx = test_indices[i]
        _, hr_s = dataset[dataset_idx]
        
        # Precip is usually at channel 3
        rain = to_mm(hr_s[PRECIP_CH].numpy(), PRECIP_MEAN, PRECIP_STD)
        
        # Clip to Singapore region to find max intensity locally
        rain_sg, _, _ = clip_field(rain, lat_hr_full, lon_hr_full, clip_ext)
        
        if rain_sg.size > 0:
            v = float(np.nanmax(rain_sg))
            if v > WET_THRESH_MM:
                sg_candidates.append((v, dataset_idx))

    sg_candidates.sort(reverse=True)
    top_samples = sg_candidates[:10]

    # -- Map generation --------------------------------------------------------
    for rank, (best_val, best_idx) in enumerate(top_samples):
        date_str = idx_to_date(best_idx, time_array)
        tag = f"rank{rank+1}_{date_str}"
        print(f"\n--- Processing #{rank+1} {date_str} | Max SG Rain: {best_val:.2f} mm ---")

        lr_s, hr_s = dataset[best_idx]
        lr_t = lr_s.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Generate 2km prediction
            s2 = model.sample(lr_t, n_steps=N_INFER, embedding_fw=embedding_fw)

        # Convert both to mm/day
        gt_mm = to_mm(hr_s[PRECIP_CH].numpy(), PRECIP_MEAN, PRECIP_STD)
        s2_mm = to_mm(s2[0, PRECIP_CH].cpu().numpy(), PRECIP_MEAN, PRECIP_STD)

        # --- IMPORTANT: CLIPPING FOR THE TWO DIFFERENT FIGURES ---

        # 1. Full Domain Figure (Regional)
        # We pass the full arrays and the full_ext
        fig_full = build_map_fig(gt_mm, s2_mm, full_ext, shp, world_gdf, title_tag=tag)
        _save(fig_full, os.path.join(OUTPUT_DIR, f"map_full_{tag}.png"))

        # 2. Singapore Focused Figure (Zoomed)
        # We clip the data arrays first so the pixel grid is calculated correctly
        gt_sg, _, _ = clip_field(gt_mm, lat_hr_full, lon_hr_full, clip_ext)
        s2_sg, _, _ = clip_field(s2_mm, lat_hr_full, lon_hr_full, clip_ext)
        
        # Check for model collapse/zero prediction
        print(f"      Full Field Max - GT: {gt_mm.max():.2f}, Pred: {s2_mm.max():.2f}")
        print(f"      Singapore Max  - GT: {gt_sg.max():.2f}, Pred: {s2_sg.max():.2f}")

        fig_sg = build_sg_map_fig(gt_sg, s2_sg, clip_ext, shp, world_gdf, title_tag=tag)
        _save(fig_sg, os.path.join(OUTPUT_DIR, f"map_sg_{tag}.png"))

    print("\nAll plots generated in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
