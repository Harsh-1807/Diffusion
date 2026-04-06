# -*- coding: utf-8 -*-
# EvalCorrDiff_Singapore_FIXED.py
#
# FIXES:
#   [F1]  make_latlon: derives HR shape from actual model output, not hardcoded
#   [F2]  to_mm: applied correctly to normalized residual fields
#   [F3]  Lat orientation: consistent orientation BEFORE clipping
#   [F4]  load_geo: extracts lat/lon from ACTUAL files (not hardcoded 256,276)
#   [F5]  Shapefile loading: robust error reporting & fallback paths
#   [F6]  Global gridding: accurate 1:1 correspondence between grid points

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
from matplotlib.colors import to_rgba
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import geopandas as gpd
import xarray as xr
import pandas as pd
from shapely.geometry import Point, box

from Dataset import ClimateDataset
from Network import DiffusionUNet
from Regressor import ClimateRegressor

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

SGP_SHP = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_SGP_0.shp"
MYS_SHP = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_MYS_0.shp"
IDN_SHP = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_IDN_0.shp"
THA_SHP="/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_THA_0.shp"
OUTPUT_DIR = "./plot"
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

        lo = 0.0 if vmin is None else float(vmin)
        bounds = np.concatenate(([0], np.arange(step, vmax + step, step)))
        if bounds[-1] < vmax:
            bounds = np.append(bounds, bounds[-1] + step)
        bounds = np.round(bounds, 6)
        base_cmap = plt.get_cmap("jet", len(bounds) - 1)
        colors = base_cmap(np.arange(base_cmap.N))
        colors[0] = [1, 1, 1, 1]
        colors[1] = to_rgba("#6baed6")
        colors[2] = to_rgba("#3182bd")
        cmap = ListedColormap(colors)

    else:
        lo = -vmax if vmin is None else float(vmin)
        hi = +vmax
        bounds = np.linspace(lo, hi, n_levels + 1)
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
# GEO - ROBUST SHAPEFILE & COORDINATE LOADING
# =============================================================================

def load_geo():
    """
    [F4] Extract lat/lon from ACTUAL data files (not hardcoded).
    [F5] Load shapefiles with detailed error reporting.
    """
    lat = None
    lon = None
    
    # Extract lat/lon from ACTUAL data files
    print("  [F4] Extracting lat/lon from data files...")
    for file_path in LR_FILES:
        if not os.path.exists(file_path):
            print(f"      WARNING: {file_path} does not exist, skipping")
            continue
        try:
            ds = xr.open_dataset(file_path)
            lat_name = next((c for c in ["lat","latitude","LAT","LATITUDE","y"]
                             if c in ds.coords or c in ds.dims), None)
            lon_name = next((c for c in ["lon","longitude","LON","LONGITUDE","x"]
                             if c in ds.coords or c in ds.dims), None)
            
            if lat_name and lon_name:
                lat = ds[lat_name].values.astype(float)
                lon = ds[lon_name].values.astype(float)
                print(f"      ✓ Found in {os.path.basename(file_path)}")
                print(f"        lat: shape={lat.shape}, range=[{lat.min():.3f}, {lat.max():.3f}]")
                print(f"        lon: shape={lon.shape}, range=[{lon.min():.3f}, {lon.max():.3f}]")
                ds.close()
                break
            ds.close()
        except Exception as e:
            print(f"      ✗ Error reading {os.path.basename(file_path)}: {e}")
            continue

    # Fallback: use defaults for SE Asia domain
    if lat is None or lon is None:
        print("  WARNING: Could not extract from files, using default SE Asia domain")
        lat = np.linspace(-2.0, 4.5, 256)
        lon = np.linspace(100.5, 107.0, 276)
        print(f"    lat: shape={lat.shape}, range=[{lat.min():.3f}, {lat.max():.3f}]")
        print(f"    lon: shape={lon.shape}, range=[{lon.min():.3f}, {lon.max():.3f}]")

    # [F5] Load Singapore boundary with error reporting
    shp = None
    print("  [F5] Loading Singapore boundary shapefile...")
    if os.path.exists(SHAPEFILE):
        print(f"      Path exists: {SHAPEFILE}")
        try:
            shp = gpd.read_file(SHAPEFILE)
            if shp.crs is None:
                shp = shp.set_crs("EPSG:4326")
            else:
                shp = shp.to_crs("EPSG:4326")
            print(f"      ✓ Loaded successfully")
            print(f"        Bounds: {shp.total_bounds}")
            print(f"        CRS: {shp.crs}")
        except Exception as e:
            print(f"      ✗ Failed to load: {type(e).__name__}: {e}")
            shp = None
    else:
        print(f"      ✗ Path does not exist!")
        parent = os.path.dirname(SHAPEFILE)
        print(f"      Parent dir {parent} exists: {os.path.exists(parent)}")
        if os.path.exists(parent):
            print(f"      Files in {os.path.basename(parent)}:")
            for f in os.listdir(parent):
                print(f"        - {f}")

    # [F5] Load GADM country boundaries with error reporting
    print("  [F5] Loading GADM country boundaries...")
    world = None
    gadm_paths = [("SGP", SGP_SHP), ("MYS", MYS_SHP), ("IDN", IDN_SHP), ("THA", THA_SHP)]
    loaded_gdfs = []
    
    for country_code, shp_path in gadm_paths:
        if not os.path.exists(shp_path):
            print(f"      ✗ GADM {country_code} NOT FOUND: {shp_path}")
            parent = os.path.dirname(shp_path)
            if os.path.exists(parent):
                print(f"        Available files in {os.path.basename(parent)}:")
                for f in os.listdir(parent)[:5]:
                    print(f"          - {f}")
            continue
        try:
            gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
            loaded_gdfs.append(gdf)
            print(f"      ✓ Loaded GADM {country_code}: bounds={gdf.total_bounds}")
        except Exception as e:
            print(f"      ✗ Failed to load GADM {country_code}: {type(e).__name__}: {e}")

    if loaded_gdfs:
        world = pd.concat(loaded_gdfs, ignore_index=True)
        
        # Clip to your domain
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())
        
        bbox = box(lon_min - 0.5, lat_min - 0.5, lon_max + 0.5, lat_max + 0.5)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
        
        try:
            world = gpd.clip(world, bbox_gdf)
            print(f"      ✓ Clipped GADM to domain: {len(world)} features")
        except Exception as e:
            print(f"      WARNING: Could not clip GADM: {e}")
    else:
        print(f"      ✗ No GADM shapefiles loaded")

    # Build extents
    full_ext = [float(lon.min()), float(lon.max()),
                float(lat.min()), float(lat.max())]
    clip_ext = [SG_LON_MIN, SG_LON_MAX, SG_LAT_MIN, SG_LAT_MAX]

    print(f"  Full domain:     lon=[{full_ext[0]:.2f}, {full_ext[1]:.2f}], lat=[{full_ext[2]:.2f}, {full_ext[3]:.2f}]")
    print(f"  Singapore clip:  lon=[{clip_ext[0]:.2f}, {clip_ext[1]:.2f}], lat=[{clip_ext[2]:.2f}, {clip_ext[3]:.2f}]")

    return lat, lon, full_ext, clip_ext, shp, world


def clip_field(field, lat, lon, extent):
    """
    [F6] Clip a 2D field to a lat/lon extent with accurate 1:1 grid mapping.
    Assumes lat is ASCENDING (south-to-north) and field row 0 = southernmost.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    
    # Find indices
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        print(f"    WARNING: clip_field found no points in extent {extent}")
        print(f"             lat range in data: [{lat.min():.3f}, {lat.max():.3f}]")
        print(f"             lon range in data: [{lon.min():.3f}, {lon.max():.3f}]")
        return field, lat, lon
    
    r0, r1 = lat_idx[0], lat_idx[-1] + 1
    c0, c1 = lon_idx[0], lon_idx[-1] + 1
    
    # Check grid accuracy
    lat_clipped = lat[r0:r1]
    lon_clipped = lon[c0:c1]
    
    print(f"    Clipped grid: lat {len(lat_clipped)} pts [{lat_clipped[0]:.3f} to {lat_clipped[-1]:.3f}], "
          f"lon {len(lon_clipped)} pts [{lon_clipped[0]:.3f} to {lon_clipped[-1]:.3f}]")
    
    return field[r0:r1, c0:c1], lat_clipped, lon_clipped


def orient_field(field_2d, lat_arr):
    """
    [F3] Ensure field_2d rows are in ascending lat order (south at row 0).
    Returns (oriented_field, ascending_lat).
    """
    if lat_arr[0] > lat_arr[-1]:
        return field_2d[::-1, :].copy(), lat_arr[::-1].copy()
    return field_2d, lat_arr


# =============================================================================
# [F2] NORMALISATION HELPERS
# =============================================================================

def to_mm(arr_norm, mean, std):
    """
    Invert log1p normalisation: x_mm = expm1(arr_norm * std + mean).
    arr_norm is NORMALIZED (what dataset stores / model outputs).
    """
    x_log = arr_norm * std + mean
    return np.expm1(x_log).clip(0)  # precipitation cannot be negative


def debug_normalization(dataset, idx, precip_mean, precip_std):
    """
    [F2] Diagnostic: verify normalization inversion is correct.
    """
    _, hr_s = dataset[idx]
    raw_norm = hr_s[PRECIP_CH].numpy()
    mm_vals  = to_mm(raw_norm, precip_mean, precip_std)
    print(f"  [NORM DEBUG] raw_norm: min={raw_norm.min():.4f} max={raw_norm.max():.4f} "
          f"mean={raw_norm.mean():.4f}")
    print(f"  [NORM DEBUG]   mm    : min={mm_vals.min():.4f}  max={mm_vals.max():.4f}  "
          f"mean={mm_vals.mean():.4f}")
    return mm_vals


# =============================================================================
# [F1] LAT/LON GRID BUILDER - ACCURATE 1:1 MAPPING
# =============================================================================

def make_latlon(hr_shape, lat_lr, lon_lr):
    """
    [F1] Build HR lat/lon grids from ACTUAL model output shape.
    
    Ensures 1:1 correspondence between field array indices and lat/lon values.
    
    Parameters
    ----------
    hr_shape : (H, W) of the actual HR field
    lat_lr   : 1-D lat array (may be ascending or descending)
    lon_lr   : 1-D lon array
    
    Returns
    -------
    lat_hr, lon_hr : 1-D arrays, lat_hr ALWAYS ASCENDING (south-to-north)
    """
    H, W = hr_shape
    
    # Get min/max regardless of order
    lat_min = float(min(lat_lr.min(), lat_lr.max()))
    lat_max = float(max(lat_lr.min(), lat_lr.max()))
    lon_min = float(min(lon_lr.min(), lon_lr.max()))
    lon_max = float(max(lon_lr.min(), lon_lr.max()))

    # Create ASCENDING lat (always south-to-north)
    lat_hr = np.linspace(lat_min, lat_max, H)
    lon_hr = np.linspace(lon_min, lon_max, W)
    
    print(f"    make_latlon: H={H}, W={W}")
    print(f"      lat: [{lat_hr[0]:.3f}, {lat_hr[-1]:.3f}] ({len(lat_hr)} pts)")
    print(f"      lon: [{lon_hr[0]:.3f}, {lon_hr[-1]:.3f}] ({len(lon_hr)} pts)")
    
    return lat_hr, lon_hr


# =============================================================================
# EMBEDDING & MODEL
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
# PLOTTING HELPERS
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
        "font.size":        24,
        "axes.titlesize":   22,
        "axes.labelsize":   20,
    })


def _pixel_grid(ax, extent, km_spacing, color="#bdbdbd"):
    lon_min, lon_max, lat_min, lat_max = extent

    # --- convert km → degrees ---
    mean_lat = 0.5 * (lat_min + lat_max)

    deg_lat = km_spacing / 111.0
    deg_lon = km_spacing / (111.0 * np.cos(np.deg2rad(mean_lat)))

    # --- generate grid lines ---
    lat_lines = np.arange(lat_min, lat_max + deg_lat, deg_lat)
    lon_lines = np.arange(lon_min, lon_max + deg_lon, deg_lon)

    # --- draw ---
    for y in lat_lines:
        ax.axhline(y, color=color, linewidth=0.2, alpha=0.3, zorder=5)

    for x in lon_lines:
        ax.axvline(x, color=color, linewidth=0.2, alpha=0.3, zorder=5)

def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {path}")


def build_map_fig(gt, s2, extent, shp, world_gdf=None, title_tag=""):
    rain_vmax = float(np.nanpercentile(gt, 99.5))
    rain_vmax = max(rain_vmax, 1.0)
    cmap_r, norm_r, bounds_r = make_dynamic_cmap(rain_vmax, n_levels=10, kind="rain")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
    data = [gt, s2]
    titles = ["Ground Truth (8km)", "Model (2km)"]

    for i, ax in enumerate(axes):
        curr_data = data[i]   # DEFINE THIS FIRST
    
        im = ax.imshow(curr_data, origin="lower", extent=extent,
                       cmap=cmap_r, norm=norm_r, interpolation="bilinear")
    
        if world_gdf is not None:
            if "GID_0" in world_gdf.columns:
                world_no_sg = world_gdf[world_gdf["GID_0"] != "SGP"]
            else:
                world_no_sg = world_gdf  # fallback
        
            world_no_sg.boundary.plot(
                ax=ax,
                color="black",
                linewidth=0.6,
                alpha=0.6,
                zorder=9
            )
        
        # --- Plot Singapore ONLY once (clean) ---
        if shp is not None:
            shp.boundary.plot(
                ax=ax,
                color="#333333",   
                linewidth=1.5,
                alpha=0.9,
                zorder=10
            )

    
        if i == 0:
            _pixel_grid(ax, extent, km_spacing=8)   # GT 8 km  
        else:
            _pixel_grid(ax, extent, km_spacing=2)   # Model  2 km
    
        xticks = np.linspace(extent[0], extent[1], 5)
        yticks = np.linspace(extent[2], extent[3], 5)
        
        # Ensure exact boundaries are included
        xticks[0], xticks[-1] = extent[0], extent[1]
        yticks[0], yticks[-1] = extent[2], extent[3]
        
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(titles[i], fontsize=13, weight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=12)
    
        if i == 0:
            ax.set_ylabel("Latitude (°N)", fontsize=12)
    
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("mm/day", fontsize=12)

    fig.suptitle(f"Domain Comparison: {title_tag}", fontsize=22, weight="bold")
    plt.tight_layout()
    return fig


def plot_rainfall_histograms(gt_mm, pred_mm, tag, save_dir):
    """
    Improved rainfall histogram:
    - Log-scale (captures heavy tail)
    - Same bins for fair comparison
    - Handles extreme outliers robustly
    - Cleaner visuals
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------- CLEAN ----------------
    gt = gt_mm.flatten()
    pred = pred_mm.flatten()

    gt = gt[np.isfinite(gt)]
    pred = pred[np.isfinite(pred)]

    # Remove zeros (dominates distribution)
    gt = gt[gt > 0]
    pred = pred[pred > 0]

    if len(gt) == 0 or len(pred) == 0:
        print("⚠️ Empty data for histogram")
        return


    max_val = max(
        np.percentile(gt, 99.5),
        np.percentile(pred, 99.5)
    )

    # ---------------- LOG BINS ----------------
    bins = np.logspace(
        np.log10(0.01),   # minimum rainfall
        np.log10(max_val + 1e-6),
        80
    )

 
    fig1 = plt.figure(figsize=(10, 6))

    plt.hist(gt, bins=bins)
    plt.xscale("log")

    plt.title("Rainfall Distribution (8km Ground Truth)")
    plt.xlabel("Rainfall (mm/day) [log scale]")
    plt.ylabel("Frequency")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    path1 = os.path.join(save_dir, "hist_gt.png")
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved  {path1}")


    fig2 = plt.figure(figsize=(10, 6))

    plt.hist(pred, bins=bins)
    plt.xscale("log")

    plt.title("Rainfall Distribution (2km Prediction)")
    plt.xlabel("Rainfall (mm/day) [log scale]")
    plt.ylabel("Frequency")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    path2 = os.path.join(save_dir, "hist_pred.png")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved  {path2}")


# =============================================================================
# MAIN
# =============================================================================

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
    a = str(pd.to_datetime(time_array[idx]).date())
    return "-".join(a.split("-")[::-1])

def main():
    apply_style()
    time_array = load_time()

    print("Loading dataset...")
    dataset = ClimateDataset(LR_FILES, LR_FILES, rank=0)

    global PRECIP_MEAN, PRECIP_STD
    PRECIP_MEAN = dataset.mean["precip"]
    PRECIP_STD  = dataset.std["precip"]
    print(f"  PRECIP_MEAN={PRECIP_MEAN:.4f}  PRECIP_STD={PRECIP_STD:.4f}")

    # Data split
    n = len(dataset)
    gen = torch.Generator().manual_seed(42)
    _, _, test_set = random_split(
        dataset,
        [int(0.65*n), int(0.15*n), n - int(0.65*n) - int(0.15*n)],
        generator=gen,
    )
    test_indices = test_set.indices

    # Geo (with robust loading)
    print("Loading geo...")
    lat, lon, full_ext, clip_ext, shp, world_gdf = load_geo()

    # Lat orientation flag
    lat_is_descending = lat[0] > lat[-1]
    print(f"  Raw lat[0]={lat[0]:.3f} lat[-1]={lat[-1]:.3f}  descending={lat_is_descending}")

    WET_THRESH_MM = 0.01

    # Model
    print("Loading model...")
    model, embedding_fw = load_model(DEVICE)

    # Normalization check
    print("\n[NORM CHECK] Verifying normalization on sample index 0...")
    debug_normalization(dataset, test_indices[0], PRECIP_MEAN, PRECIP_STD)

    print("\nSearching for top rainfall samples over Singapore...")
    sg_candidates = []

    for i in range(len(test_set)):
        dataset_idx = test_indices[i]
        _, hr_s = dataset[dataset_idx]
        rain_norm = hr_s[PRECIP_CH].numpy()
        rain_mm = to_mm(rain_norm, PRECIP_MEAN, PRECIP_STD)

        # [F1] Build HR lat/lon from ACTUAL field shape
        hr_shape = rain_mm.shape
        lat_hr, lon_hr = make_latlon(hr_shape, lat, lon)

        # [F3] Orient field if lat was descending
        if lat_is_descending:
            rain_mm = rain_mm[::-1, :].copy()

        rain_sg, _, _ = clip_field(rain_mm, lat_hr, lon_hr, clip_ext)

        if rain_sg.size > 0:
            v = float(np.nanmax(rain_sg))
            if v > WET_THRESH_MM:
                sg_candidates.append((v, dataset_idx))

    sg_candidates.sort(reverse=True)
    top_samples = sg_candidates[:10]
    print(f"  Found {len(sg_candidates)} wet samples; using top {len(top_samples)}")

    # Map generation
    for rank, (best_val, best_idx) in enumerate(top_samples):
        date_str = idx_to_date(best_idx, time_array)
        tag = f"rank{rank+1}_{date_str}"
        sample_dir = os.path.join(OUTPUT_DIR, tag)
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\n--- Processing #{rank+1} {date_str} | Max SG Rain: {best_val:.2f} mm ---")

        lr_s, hr_s = dataset[best_idx]
        lr_t = lr_s.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            s2 = model.sample(lr_t, n_steps=N_INFER, embedding_fw=embedding_fw)

        # [F2] Convert normalized to mm
        gt_norm = hr_s[PRECIP_CH].numpy()
        s2_norm = s2[0, PRECIP_CH].cpu().numpy()
        gt_mm = to_mm(gt_norm, PRECIP_MEAN, PRECIP_STD)
        s2_mm = to_mm(s2_norm, PRECIP_MEAN, PRECIP_STD)

        print(f"  Normalized  - GT: [{gt_norm.min():.3f}, {gt_norm.max():.3f}]  "
              f"Pred: [{s2_norm.min():.3f}, {s2_norm.max():.3f}]")
        print(f"  mm/day      - GT: [{gt_mm.min():.2f}, {gt_mm.max():.2f}]  "
              f"Pred: [{s2_mm.min():.2f}, {s2_mm.max():.2f}]")

        # [F1] Build HR lat/lon from actual shapes
        lat_hr, lon_hr = make_latlon(gt_mm.shape, lat, lon)
        lat_hr_s2, lon_hr_s2 = make_latlon(s2_mm.shape, lat, lon)

        # [F3] Orient consistently
        if lat_is_descending:
            gt_mm = gt_mm[::-1, :].copy()
            s2_mm = s2_mm[::-1, :].copy()

        # Full domain figure
        fig_full = build_map_fig(gt_mm, s2_mm, full_ext, shp, world_gdf, title_tag=tag)  
        _save(fig_full, os.path.join(sample_dir, "map_full.png"))

        # Singapore zoomed figure
        print(f"  Clipping GT...")
        gt_sg, _, _ = clip_field(gt_mm, lat_hr, lon_hr, clip_ext)
        print(f"  Clipping S2...")
        s2_sg, _, _ = clip_field(s2_mm, lat_hr_s2, lon_hr_s2, clip_ext)

        print(f"  Singapore   - GT: [{gt_sg.min():.2f}, {gt_sg.max():.2f}]  "
              f"Pred: [{s2_sg.min():.2f}, {s2_sg.max():.2f}]")

        if gt_sg.size == 0 or s2_sg.size == 0:
            print("  WARNING: Singapore clip returned empty array!")
            continue

        fig_sg = build_map_fig(gt_sg, s2_sg, clip_ext, shp, world_gdf, title_tag=tag)
        _save(fig_sg, os.path.join(sample_dir, "map_sg.png"))
        plot_rainfall_histograms(gt_mm, s2_mm, tag, sample_dir)

    print("\nAll plots generated in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
