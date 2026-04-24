# -*- coding: utf-8 -*-
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
from tqdm import tqdm
import pandas as pd
from shapely.geometry import Point, box

from Dataset import ClimateDataset
from Network import DiffusionUNet
from scipy.ndimage import gaussian_filter
from Regressor import ClimateRegressor

# =============================================================================
# PATHS
# =============================================================================
'''LR_FILES = [
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/gt_extracted/huss_rcm_8km_daily_data_2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/gt_extracted/mslp_rcm_8km_daily_data_2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/gt_extracted/tas_rcm_8km_daily_data_2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/gt_extracted/precip_rcm_8km_daily_data_2014.nc",
]'''

LR_FILES = [
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/huss_rcm_8km_daily_data_1995-2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/mslp_rcm_8km_daily_data_1995-2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/tas_rcm_8km_daily_data_1995-2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/precip_rcm_8km_daily_data_1995-2014.nc",
]


BEST_CKPT = (
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/CorrDiff/2km/"
    "corrdiff_singapore_v7b_latest.pth"
)
REG_CKPT  = "regressor.pth"

SHAPEFILE = (
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/"
    "ShapeFiles/singapore_Singapore_Country_Boundary.shp"
)
SGP_SHP = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_SGP_0.shp"
MYS_SHP = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_MYS_0.shp"
IDN_SHP = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_IDN_0.shp"
THA_SHP = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_THA_0.shp"

OUTPUT_DIR = "./plot_main_v2"

# [G1] Actual 2km Ground Truth files
GT_2KM_FILES = [
   
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/Ground_Truth/precip_rcm_2km_daily_data_2005.nc",
    
]
# Variable name inside the 2km GT files (change if different)
GT_2KM_VAR = "pr"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRECIP_CH   = 3          # channel index in dataset tensor for precipitation
N_CH        = 4
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
# [G1]  2-KM GT LOADER
# =============================================================================

class GT2kmLoader:
    """
    Loads actual 2km ground-truth precipitation from one or more NetCDF files.
    Builds a time-indexed lookup so you can query by date string (YYYY-MM-DD).

    Usage:
        loader = GT2kmLoader(GT_2KM_FILES, var_name="pr")
        loader.load()
        field_2d, lat, lon = loader.get_by_date("2014-07-15")
    """

    def __init__(self, nc_files, var_name=GT_2KM_VAR):
        self.nc_files = [f for f in nc_files if os.path.exists(f)]
        self.var_name = var_name
        self._ds = None          # concatenated xarray dataset
        self._lat = None
        self._lon = None
        self._times_str = None   # list of "YYYY-MM-DD" strings, same length as time axis
        self._loaded = False

    # ------------------------------------------------------------------
    def load(self):
        if not self.nc_files:
            print("  [GT2km] WARNING: No 2km GT files found. Paths checked:")
            for f in GT_2KM_FILES:
                print(f"    {f}  exists={os.path.exists(f)}")
            return

        print(f"  [GT2km] Loading {len(self.nc_files)} file(s)...")
        datasets = []
        for f in self.nc_files:
            try:
                ds = xr.open_dataset(f)
                print(f"    ✓ {os.path.basename(f)}")
                print(f"      vars: {list(ds.data_vars)}   coords: {list(ds.coords)}")
                datasets.append(ds)
            except Exception as e:
                print(f"    ✗ {os.path.basename(f)}: {e}")

        if not datasets:
            print("  [GT2km] No datasets loaded.")
            return

        # Concatenate along time
        self._ds = xr.concat(datasets, dim="time")
        self._ds = self._ds.sortby("time")

        # Extract lat/lon
        lat_name = next((c for c in ["lat","latitude","LAT","y"] if c in self._ds.coords), None)
        lon_name = next((c for c in ["lon","longitude","LON","x"] if c in self._ds.coords), None)

        if lat_name is None or lon_name is None:
            print(f"  [GT2km] Could not find lat/lon. Available coords: {list(self._ds.coords)}")
            return

        self._lat = self._ds[lat_name].values.astype(float)
        self._lon = self._ds[lon_name].values.astype(float)

        # Build date-string index
        times_np = self._ds["time"].values
        self._times_str = [str(pd.to_datetime(t).date()) for t in times_np]
        self._time_index = {d: i for i, d in enumerate(self._times_str)}  # O(1) lookup, no duplicates

        print(f"  [GT2km] Total time steps: {len(self._times_str)}")
        print(f"          Date range: {self._times_str[0]} → {self._times_str[-1]}")
        print(f"          lat: {self._lat.shape}  [{self._lat.min():.3f}, {self._lat.max():.3f}]")
        print(f"          lon: {self._lon.shape}  [{self._lon.min():.3f}, {self._lon.max():.3f}]")

        # Detect variable
        if self.var_name not in self._ds.data_vars:
            available = list(self._ds.data_vars)
            print(f"  [GT2km] Variable '{self.var_name}' not found. Available: {available}")
            # Try to auto-detect precipitation variable
            precip_candidates = [v for v in available
                                  if any(k in v.lower() for k in ["pr","precip","rain","prec"])]
            if precip_candidates:
                self.var_name = precip_candidates[0]
                print(f"  [GT2km] Auto-detected var: '{self.var_name}'")
            else:
                self.var_name = available[0]
                print(f"  [GT2km] Falling back to first var: '{self.var_name}'")

        self._loaded = True

    # ------------------------------------------------------------------
    def get_by_date(self, date_str):
        """
        Returns (field_2d, lat_1d, lon_1d) for a given "YYYY-MM-DD" string.
        field_2d is in ASCENDING lat order (south at row 0).
        Returns (None, None, None) if date not found.
        """
        if not self._loaded:
            return None, None, None

        if date_str not in self._times_str:
            return None, None, None

        t_idx = self._time_index[date_str]

        field = self._ds[self.var_name].isel(time=t_idx).values.astype(float)

        # Ensure 2D
        if field.ndim != 2:
            field = np.squeeze(field)

        # Ensure ascending lat (south at row 0)
        lat = self._lat.copy()
        lon = self._lon.copy()
        if lat[0] > lat[-1]:
            field = field[::-1, :].copy()
            lat   = lat[::-1].copy()

        return field, lat, lon

    # ------------------------------------------------------------------
    def has_date(self, date_str):
        return self._loaded and date_str in self._time_index

    # ------------------------------------------------------------------
    @property
    def lat(self):
        return self._lat

    @property
    def lon(self):
        return self._lon

    @property
    def all_dates(self):
        return self._times_str if self._times_str else []


# =============================================================================
# [G3]  CORRELATION CHECK: 8km GT vs 2km GT
# =============================================================================

def check_8km_vs_2km_correlation(dataset, gt2km_loader, test_indices, time_array,
                                  lat_lr, lon_lr, precip_mean, precip_std,
                                  n_samples=30):
    """
    [G3] Verifies that 8km dataset GT and 2km actual GT are well-correlated.
    Uses spatial mean (domain-average) to compare time series.
    Prints Pearson-r and generates a scatter plot.
    """
    print("\n[G3] ===== 8km GT vs 2km GT Correlation Check =====")

    if not gt2km_loader._loaded:
        print("  SKIP: 2km GT not loaded.")
        return

    lat_is_desc = lat_lr[0] > lat_lr[-1]

    means_8km = []
    means_2km = []
    matched_dates = []

    checked = 0
    for idx in test_indices:
        if checked >= n_samples:
            break

        date_str = str(pd.to_datetime(time_array[idx]).date())

        if not gt2km_loader.has_date(date_str):
            continue

        # ---- 8km GT ----
        _, hr_s = dataset[idx]
        gt8_norm = hr_s[PRECIP_CH].numpy()
        gt8_mm   = to_mm(gt8_norm, precip_mean, precip_std)

        if lat_is_desc:
            gt8_mm = gt8_mm[::-1, :]

        # Clip to Singapore region for fair comparison
        lat_hr8, lon_hr8 = make_latlon(gt8_mm.shape, lat_lr, lon_lr)
        gt8_sg, _, _ = clip_field(gt8_mm, lat_hr8, lon_hr8,
                                   [SG_LON_MIN, SG_LON_MAX, SG_LAT_MIN, SG_LAT_MAX])

        if gt8_sg.size == 0:
            continue

        # ---- 2km GT ----
        gt2_raw, lat2, lon2 = gt2km_loader.get_by_date(date_str)
        gt2_sg, _, _ = clip_field(gt2_raw, lat2, lon2,
                                   [SG_LON_MIN, SG_LON_MAX, SG_LAT_MIN, SG_LAT_MAX])

        if gt2_sg.size == 0:
            continue

        # Convert 2km to mm if needed (units check)
        # If values look like kg/m2/s (very small), convert to  
        gt2_mean_raw = float(np.nanmean(gt2_sg))
        gt2_sg_mm = convert_gt2km_units(gt2_sg)

        means_8km.append(float(np.nanmean(gt8_sg)))
        means_2km.append(float(np.nanmean(gt2_sg_mm)))
        matched_dates.append(date_str)
        checked += 1

    if len(means_8km) < 3:
        print(f"  WARNING: Only {len(means_8km)} matched dates found — cannot compute correlation.")
        return

    means_8km = np.array(means_8km)
    means_2km = np.array(means_2km)

    r, p = pearsonr(means_8km, means_2km)
    print(f"  Matched samples: {len(means_8km)}")
    print(f"  8km GT  — mean={means_8km.mean():.4f}  std={means_8km.std():.4f}")
    print(f"  2km GT  — mean={means_2km.mean():.4f}  std={means_2km.std():.4f}")
    print(f"  Pearson-r = {r:.4f}   p = {p:.4e}")

    if r < 0.3:
        print("  ⚠️  LOW CORRELATION: Check units / variable alignment of 2km GT files!")
    elif r > 0.7:
        print("  ✅ Good correlation — 2km GT files appear consistent with 8km GT.")
    else:
        print("  ⚠️  Moderate correlation — check units or spatial domain alignment.")

    # ---- Scatter plot ----
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    ax.scatter(means_8km, means_2km, alpha=0.7, edgecolors="k", s=60, color="#4393C3")

    # 1:1 line
    lims = [min(means_8km.min(), means_2km.min()),
            max(means_8km.max(), means_2km.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="1:1 line")

    ax.set_xlabel("8km GT ", fontsize=13)
    ax.set_ylabel("2km GT ", fontsize=13)
    ax.set_title(f"8km vs 2km GT  [r = {r:.3f}]", fontsize=14, weight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    corr_path = os.path.join(OUTPUT_DIR, "corr_8km_vs_2km_GT.png")
    fig.savefig(corr_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {corr_path}")


def convert_gt2km_units(field):
    """
    Auto-detect and convert 2km GT units to  .
    RCM files often store in kg m-2 s-1 (= mm/s).
    """
    mean_val = float(np.nanmean(np.abs(field[field > 0]))) if np.any(field > 0) else 0
    if mean_val < 0.1:
        # Likely kg/m2/s → convert to   (* 86400)
        return field * 86400.0
    return field  # Already in  


# =============================================================================
# [G2]  ACCURATE SAMPLE FINDER
# =============================================================================

def find_top_samples_using_gt2km(dataset, gt2km_loader, test_indices,
                                  time_array, lat_lr, lon_lr,
                                  precip_mean, precip_std, top_k=10):
    """
    [G2] Find heavy-rain samples that exist in BOTH:
         - the 8km dataset (test split)
         - the actual 2km GT files
    Ranks by ACTUAL 2km GT max rainfall in the Singapore region.
    Also reports 8km GT for comparison (correlation check embedded).
    """
    print("\n[G2] ===== Finding Top Samples (2km GT matched) =====")

    if not gt2km_loader._loaded:
        print("  WARNING: 2km GT not loaded — falling back to 8km-based ranking.")
        return _find_top_samples_8km(dataset, test_indices, time_array,
                                     lat_lr, lon_lr, precip_mean, precip_std, top_k)

    lat_is_desc = lat_lr[0] > lat_lr[-1]
    candidates = []

    for idx in tqdm(test_indices, desc="Scanning test set"):
        date_str = str(pd.to_datetime(time_array[idx]).date())

        # Must exist in 2km GT
        if not gt2km_loader.has_date(date_str):
            continue

        # Load 2km GT for this date
        gt2_raw, lat2, lon2 = gt2km_loader.get_by_date(date_str)
        gt2_mm = convert_gt2km_units(gt2_raw)

        gt2_sg, _, _ = clip_field(gt2_mm, lat2, lon2,
                                   [SG_LON_MIN, SG_LON_MAX, SG_LAT_MIN, SG_LAT_MAX])

        if gt2_sg.size == 0:
            continue

        max_2km = float(np.nanmax(gt2_sg))

        if max_2km < 0.01:   # skip dry days
            continue

        # Also load 8km GT for reference
        _, hr_s = dataset[idx]
        gt8_norm = hr_s[PRECIP_CH].numpy()
        gt8_mm   = to_mm(gt8_norm, precip_mean, precip_std)

        if lat_is_desc:
            gt8_mm = gt8_mm[::-1, :]

        lat_hr8, lon_hr8 = make_latlon(gt8_mm.shape, lat_lr, lon_lr)
        gt8_sg, _, _ = clip_field(gt8_mm, lat_hr8, lon_hr8,
                                   [SG_LON_MIN, SG_LON_MAX, SG_LAT_MIN, SG_LAT_MAX])
        max_8km = float(np.nanmax(gt8_sg)) if gt8_sg.size > 0 else 0.0

        candidates.append({
            "dataset_idx": idx,
            "date_str":    date_str,
            "max_2km_gt":  max_2km,
            "max_8km_gt":  max_8km,
        })

    if not candidates:
        print("  WARNING: No matched dates found between test set and 2km GT files.")
        print("  Falling back to 8km-based ranking.")
        return _find_top_samples_8km(dataset, test_indices, time_array,
                                     lat_lr, lon_lr, precip_mean, precip_std, top_k)

    # Sort by 2km GT max
    candidates.sort(key=lambda d: d["max_2km_gt"], reverse=True)
    top = candidates[:top_k]

    print(f"  Total matched dates: {len(candidates)}")
    print(f"  Top {len(top)} selected:")
    for i, c in enumerate(top):
        print(f"    #{i+1}  {c['date_str']}  "
              f"2km-max={c['max_2km_gt']:.2f}  8km-max={c['max_8km_gt']:.2f}")

    return top


def _find_top_samples_8km(dataset, test_indices, time_array,
                           lat_lr, lon_lr, precip_mean, precip_std, top_k=10):
    """Fallback: rank by 8km GT max (original behaviour)."""
    lat_is_desc = lat_lr[0] > lat_lr[-1]
    candidates = []

    for idx in tqdm(test_indices, desc="8km fallback scan"):
        _, hr_s = dataset[idx]
        rain_norm = hr_s[PRECIP_CH].numpy()
        rain_mm   = to_mm(rain_norm, precip_mean, precip_std)

        if lat_is_desc:
            rain_mm = rain_mm[::-1, :]

        lat_hr, lon_hr = make_latlon(rain_mm.shape, lat_lr, lon_lr)
        rain_sg, _, _ = clip_field(rain_mm, lat_hr, lon_hr,
                                    [SG_LON_MIN, SG_LON_MAX, SG_LAT_MIN, SG_LAT_MAX])

        if rain_sg.size > 0:
            v = float(np.nanmax(rain_sg))
            if v > 0.01:
                date_str = str(pd.to_datetime(time_array[idx]).date())
                candidates.append({"dataset_idx": idx, "date_str": date_str,
                                    "max_2km_gt": None, "max_8km_gt": v})

    candidates.sort(key=lambda d: d["max_8km_gt"], reverse=True)
    return candidates[:top_k]


# =============================================================================
# [G4]  COMMON-GRID REGRIDDER
# =============================================================================

def regrid_to_common(field_src, lat_src, lon_src, lat_tgt, lon_tgt):
    """
    [G4] Bilinear regrid of field_src (lat_src × lon_src) → (lat_tgt × lon_tgt).
    Uses torch interpolation for speed.
    Both lat arrays must be ASCENDING.
    Returns field on (lat_tgt, lon_tgt) grid.
    """
    t = torch.tensor(field_src, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    out_h = len(lat_tgt)
    out_w = len(lon_tgt)
    result = F.interpolate(t, size=(out_h, out_w),
                           mode="bilinear", align_corners=True)
    return result[0, 0].numpy()


def align_to_common_grid(gt2_mm, lat2, lon2, pred_mm, lat_pred, lon_pred):
    """
    [G4] Bring GT (2km) and model prediction to a COMMON grid.
    We pick the FINER grid as the target so we don't lose detail.
    Returns: gt_common, pred_common, lat_common, lon_common
    """
    # Choose finer target (more points)
    if len(lat2) >= len(lat_pred):
        lat_tgt, lon_tgt = lat2, lon2
        pred_common = regrid_to_common(pred_mm, lat_pred, lon_pred, lat_tgt, lon_tgt)
        gt_common   = gt2_mm
    else:
        lat_tgt, lon_tgt = lat_pred, lon_pred
        gt_common   = regrid_to_common(gt2_mm, lat2, lon2, lat_tgt, lon_tgt)
        pred_common = pred_mm

    print(f"    Common grid: {lat_tgt.shape[0]}×{lon_tgt.shape[0]}  "
          f"lat=[{lat_tgt[0]:.3f},{lat_tgt[-1]:.3f}]  "
          f"lon=[{lon_tgt[0]:.3f},{lon_tgt[-1]:.3f}]")

    return gt_common, pred_common, lat_tgt, lon_tgt


# =============================================================================
# [G5]  METRICS
# =============================================================================

def compute_metrics(gt, pred, tag=""):
    """[G5] RMSE, MAE, Bias, Pearson-r on spatially-aligned arrays."""
    assert gt.shape == pred.shape, f"Shape mismatch: {gt.shape} vs {pred.shape}"

    mask = np.isfinite(gt) & np.isfinite(pred)
    g = gt[mask].ravel()
    p = pred[mask].ravel()

    if len(g) == 0:
        return {}

    bias = float(np.mean(p - g))
    rmse = float(np.sqrt(np.mean((p - g)**2)))
    mae  = float(np.mean(np.abs(p - g)))
    r    = float(pearsonr(g, p)[0]) if len(g) > 1 else float("nan")

    print(f"  [{tag}] RMSE={rmse:.4f}  MAE={mae:.4f}  Bias={bias:.4f}  r={r:.4f}")
    return {"rmse": rmse, "mae": mae, "bias": bias, "pearson_r": r}


# =============================================================================
# COLOURMAP  (unchanged)
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
# GEO  (from v1, unchanged)
# =============================================================================

def load_geo():
    lat = None
    lon = None

    print("  [F4] Extracting lat/lon from data files...")
    for file_path in LR_FILES:
        if not os.path.exists(file_path):
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
                ds.close()
                break
            ds.close()
        except Exception:
            continue

    if lat is None or lon is None:
        lat = np.linspace(-2.0, 4.5, 256)
        lon = np.linspace(100.5, 107.0, 276)

    shp = None
    if os.path.exists(SHAPEFILE):
        try:
            shp = gpd.read_file(SHAPEFILE)
            shp = shp.set_crs("EPSG:4326") if shp.crs is None else shp.to_crs("EPSG:4326")
        except Exception as e:
            print(f"      ✗ Singapore shapefile: {e}")

    world = None
    gadm_paths = [("SGP", SGP_SHP), ("MYS", MYS_SHP), ("IDN", IDN_SHP), ("THA", THA_SHP)]
    loaded_gdfs = []
    for country_code, shp_path in gadm_paths:
        if not os.path.exists(shp_path):
            continue
        try:
            gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
            loaded_gdfs.append(gdf)
        except Exception as e:
            print(f"      ✗ GADM {country_code}: {e}")

    if loaded_gdfs:
        world = pd.concat(loaded_gdfs, ignore_index=True)
        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())
        bbox = box(lon_min-0.5, lat_min-0.5, lon_max+0.5, lat_max+0.5)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
        try:
            world = gpd.clip(world, bbox_gdf)
        except Exception:
            pass

    full_ext = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]
    clip_ext = [SG_LON_MIN, SG_LON_MAX, SG_LAT_MIN, SG_LAT_MAX]
    return lat, lon, full_ext, clip_ext, shp, world


def clip_field(field, lat, lon, extent):
    lon_min, lon_max, lat_min, lat_max = extent
    lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_idx = np.where((lon >= lon_min) & (lon <= lon_max))[0]

    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return field, lat, lon

    r0, r1 = lat_idx[0], lat_idx[-1] + 1
    c0, c1 = lon_idx[0], lon_idx[-1] + 1
    return field[r0:r1, c0:c1], lat[r0:r1], lon[c0:c1]


def orient_field(field_2d, lat_arr):
    if lat_arr[0] > lat_arr[-1]:
        return field_2d[::-1, :].copy(), lat_arr[::-1].copy()
    return field_2d, lat_arr


# =============================================================================
# NORMALISATION
# =============================================================================

def to_mm(arr_norm, mean, std):
    x_log = arr_norm * std + mean
    return np.expm1(x_log).clip(0)


def debug_normalization(dataset, idx, precip_mean, precip_std):
    _, hr_s = dataset[idx]
    raw_norm = hr_s[PRECIP_CH].numpy()
    mm_vals  = to_mm(raw_norm, precip_mean, precip_std)
    print(f"  [NORM DEBUG] raw_norm: min={raw_norm.min():.4f} max={raw_norm.max():.4f}")
    print(f"  [NORM DEBUG]   mm    : min={mm_vals.min():.4f}  max={mm_vals.max():.4f}")
    return mm_vals


# =============================================================================
# LAT/LON BUILDER
# =============================================================================

def make_latlon(hr_shape, lat_lr, lon_lr):
    H, W = hr_shape
    lat_min = float(min(lat_lr.min(), lat_lr.max()))
    lat_max = float(max(lat_lr.min(), lat_lr.max()))
    lon_min = float(min(lon_lr.min(), lon_lr.max()))
    lon_max = float(max(lon_lr.min(), lon_lr.max()))
    lat_hr  = np.linspace(lat_min, lat_max, H)
    lon_hr  = np.linspace(lon_min, lon_max, W)
    return lat_hr, lon_hr


# =============================================================================
# EMBEDDING & MODEL  (unchanged from v1)
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
        return max(5, min(int(self.s_freq * 2.0 * self.n_noise_steps), self.n_noise_steps))

    def _apply_qdm_channel(self, field_ch, ch):
        arr    = field_ch.cpu().numpy()
        mapped = np.interp(arr, self.qdm_lr[ch], self.qdm_hr[ch])
        return torch.tensor(mapped, dtype=field_ch.dtype, device=self.device)

    def g(self, lr_field, hr_size):
        if not self._fitted:
            raise RuntimeError("EmbeddingFramework not loaded.")
        B, C = lr_field.shape[:2]
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
        np.ndarray, np.dtype,
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

    embedding_fw = None
    emb_state    = ckpt.get("embedding_state", None)
    use_emb      = ckpt.get("use_embedding", False)

    if use_emb and emb_state is not None and emb_state.get("fitted", False):
        embedding_fw = EmbeddingFramework(device)
        embedding_fw.load_state_dict(emb_state)

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
        "font.size":        14,
        "axes.titlesize":   13,
        "axes.labelsize":   12,
    })


def _pixel_grid(ax, extent, km_spacing, sg_extent=None,
                color="#bdbdbd", sg_color="#5c5c5c"):
    lon_min, lon_max, lat_min, lat_max = extent
    mean_lat = 0.5 * (lat_min + lat_max)
    deg_lat  = km_spacing / 111.0
    deg_lon  = km_spacing / (111.0 * np.cos(np.deg2rad(mean_lat)))
    lat_lines = np.arange(lat_min, lat_max + deg_lat, deg_lat)
    lon_lines = np.arange(lon_min, lon_max + deg_lon, deg_lon)
    for y in lat_lines:
        ax.axhline(y, color=color, linewidth=0.2, alpha=0.3, zorder=5)
    for x in lon_lines:
        ax.axvline(x, color=color, linewidth=0.2, alpha=0.3, zorder=5)
    if sg_extent is not None:
        sg_lon_min, sg_lon_max, sg_lat_min, sg_lat_max = sg_extent
        for y in lat_lines:
            if sg_lat_min <= y <= sg_lat_max:
                ax.hlines(y, sg_lon_min, sg_lon_max,
                          color=sg_color, linewidth=0.4, alpha=0.8, zorder=6)
        for x in lon_lines:
            if sg_lon_min <= x <= sg_lon_max:
                ax.vlines(x, sg_lat_min, sg_lat_max,
                          color=sg_color, linewidth=0.4, alpha=0.8, zorder=6)


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {path}")


def _add_boundaries(ax, shp, world_gdf):
    if world_gdf is not None:
        world_no_sg = (world_gdf[world_gdf["GID_0"] != "SGP"]
                       if "GID_0" in world_gdf.columns else world_gdf)
        world_no_sg.boundary.plot(ax=ax, color="black", linewidth=0.6,
                                   alpha=0.7, zorder=9)
    if shp is not None:
        shp.boundary.plot(ax=ax, color="#000080", linewidth=1.2,
                          alpha=0.8, zorder=10)


# =============================================================================
# [G6]  THREE-PANEL COMPARISON FIGURE: 8km GT | 2km GT | Model 2km
# =============================================================================
def build_three_panel_full(gt8, gt2, pred,
                          lat8, lon8,
                          lat2, lon2,
                          lat_pred, lon_pred,
                          shp, world_gdf,
                          metrics, title_tag):

    def fix(field, lat):
        if lat[0] > lat[-1]:
            return field[::-1, :], lat[::-1]
        return field, lat

    # ---- orientation ----
    gt8, lat8 = fix(gt8, lat8)
    gt2, lat2 = fix(gt2, lat2)
    pred, lat_pred = fix(pred, lat_pred)

    # ---- MASTER EXTENT = GT2 ----
    extent = [lon2.min(), lon2.max(), lat2.min(), lat2.max()]

    # 🔴 IMPORTANT:
    # DO NOT regrid gt8 → preserve original structure
    # Only regrid prediction → GT2 grid
    pred_r = regrid_to_common(pred, lat_pred, lon_pred, lat2, lon2)

    # ---- color ----
    rain_vmax = float(np.nanpercentile(
        np.concatenate([gt8.ravel(), pred_r.ravel(), gt2.ravel()]), 99.5))
    rain_vmax = max(rain_vmax, 1.0)

    cmap_r, norm_r, _ = make_dynamic_cmap(rain_vmax, n_levels=10, kind="rain")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")

    panels = [
        (gt8,    "8KM Input"),
        (pred_r, "Predicted"),
        (gt2,    "GT Provided"),
    ]

    for i, (ax, (data, title)) in enumerate(zip(axes, panels)):

        im = ax.imshow(data, origin="lower", extent=extent,
                       cmap=cmap_r, norm=norm_r, interpolation="bilinear")

        _add_boundaries(ax, shp, world_gdf)

        # ✅ CORRECT GRID
        km_sp = 8 if i == 0 else 2
        _pixel_grid(ax, extent, km_spacing=km_sp)

        xticks = np.linspace(extent[0], extent[1], 4)
        yticks = np.linspace(extent[2], extent[3], 4)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        ax.set_title(title, fontsize=18, weight="bold")

        if i == 0:
            ax.set_ylabel("Latitude (°N)")
        ax.set_xlabel("Longitude (°E)")

    # ---- colorbar ----
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Precip Flux (kg m⁻² s⁻¹)", fontsize=11)

    # ---- metrics ----
    axes[1].text(0.03, 0.97,
                 f"r={metrics.get('pearson_r', float('nan')):.3f}\n"
                 f"RMSE={metrics.get('rmse', float('nan')):.2f}",
                 transform=axes[1].transAxes,
                 fontsize=10, va="top", ha="left",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle(f"{title_tag}", fontsize=15, weight="bold")
    

 
    return fig
def build_three_panel(gt8_sg, gt2_common, pred_common,
                      lat_gt2, lon_gt2,
                      shp, world_gdf,
                      metrics_gt2_vs_pred, title_tag):

    extent = [lon_gt2.min(), lon_gt2.max(),
              lat_gt2.min(), lat_gt2.max()]

    # 🔴 DO NOT TOUCH gt8_sg (keep 8km structure)
    pred_r = regrid_to_common(pred_common, lat_gt2, lon_gt2, lat_gt2, lon_gt2)

    rain_vmax = float(np.nanpercentile(
        np.concatenate([gt8_sg.ravel(), pred_r.ravel(), gt2_common.ravel()]), 99.5))
    rain_vmax = max(rain_vmax, 1.0)

    cmap_r, norm_r, _ = make_dynamic_cmap(rain_vmax, n_levels=10, kind="rain")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")

    panels = [
        (gt8_sg,     "8KM Input"),      # ← FIXED
        (pred_r,     "Predicted"),
        (gt2_common, "GT Provided"),    # ← FIXED
    ]


    for i, (ax, (data, title)) in enumerate(zip(axes, panels)):

        im = ax.imshow(data, origin="lower", extent=extent,
                       cmap=cmap_r, norm=norm_r, interpolation="bilinear")

        _add_boundaries(ax, shp, world_gdf)

        # ✅ CORRECT GRID
        km_sp = 8 if i == 0 else 2
        _pixel_grid(ax, extent, km_spacing=km_sp, sg_extent=extent)

        xticks = np.linspace(extent[0], extent[1], 4)
        yticks = np.linspace(extent[2], extent[3], 4)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        ax.set_title(title, fontsize=18, weight="bold")

        if i == 0:
            ax.set_ylabel("Latitude (°N)")
        ax.set_xlabel("Longitude (°E)")

    # ---- colorbar ----
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Precip Flux (kg m⁻² s⁻¹)", fontsize=11)

    # ---- metrics ----
    axes[1].text(0.03, 0.97,
                 f"r={metrics_gt2_vs_pred.get('pearson_r', float('nan')):.3f}\n"
                 f"RMSE={metrics_gt2_vs_pred.get('rmse', float('nan')):.2f}",
                 transform=axes[1].transAxes,
                 fontsize=10, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="white", edgecolor="gray", alpha=0.8))

    fig.suptitle(f"{title_tag}", fontsize=15, weight="bold")
    


    return fig


def plot_scatter(gt_common, pred_common, metrics, tag, save_dir):
    g = gt_common.ravel()
    p = pred_common.ravel()
    mask = np.isfinite(g) & np.isfinite(p) & (g > 0) & (p > 0)
    g, p = g[mask], p[mask]
    if len(g) < 10:
        return

    # Subsample for speed
    if len(g) > 5000:
        sel = np.random.choice(len(g), 5000, replace=False)
        g, p = g[sel], p[sel]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    try:
        xy   = np.vstack([g, p])
        kde  = gaussian_kde(xy)
        dens = kde(xy)
        sc   = ax.scatter(g, p, c=dens, cmap="plasma", s=10, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="KDE density")
    except Exception:
        ax.scatter(g, p, alpha=0.3, s=10, color="#4393C3")

    lims = [min(g.min(), p.min()), max(g.max(), p.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="1:1 line")

    r_val   = metrics.get("pearson_r", float("nan"))
    rmse_val = metrics.get("rmse", float("nan"))
    ax.set_xlabel("2km GT ( )", fontsize=12)
    ax.set_ylabel("Model prediction ", fontsize=12)
    ax.set_title(f"Scatter  r={r_val:.3f}  RMSE={rmse_val:.2f}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "scatter_gt2km_vs_model.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_rainfall_histograms(gt_mm, pred_mm, tag, save_dir):
    for data, name, fname in [
        (gt_mm,   "2km GT",          "hist_gt2km.png"),
        (pred_mm, "Model Prediction", "hist_pred2km.png"),
    ]:
        vals = data.flatten()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) == 0:
            continue
        max_val = np.percentile(vals, 99.5)
        bins = np.logspace(np.log10(0.01), np.log10(max_val + 1e-6), 80)
        fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
        ax.hist(vals, bins=bins, color="#4393C3", edgecolor="none")
        ax.set_xscale("log")
        ax.set_title(f"Rainfall Distribution — {name}")
        ax.set_xlabel("Precipitation  [log]")
        ax.set_ylabel("Frequency")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {os.path.join(save_dir, fname)}")


# =============================================================================
# TIME LOADER
# =============================================================================

def load_time():
    ds = xr.open_dataset(LR_FILES[3])
    time_name = next((c for c in ["time","TIME","date"]
                      if c in ds.coords or c in ds.dims), None)
    if time_name is None:
        raise RuntimeError("No time coordinate found in dataset")
    t = ds[time_name].values
    ds.close()
    return t


def idx_to_date(idx, time_array):
    return str(pd.to_datetime(time_array[idx]).date())


# =============================================================================
# MAIN
# =============================================================================
from scipy.ndimage import gaussian_filter

def main():
    apply_style()
    time_array = load_time()

    print("Loading dataset...")
    dataset = ClimateDataset(LR_FILES, LR_FILES, rank=0)

    global PRECIP_MEAN, PRECIP_STD
    PRECIP_MEAN = dataset.mean["precip"]
    PRECIP_STD  = dataset.std["precip"]
    print(f"  PRECIP_MEAN={PRECIP_MEAN:.4f}  PRECIP_STD={PRECIP_STD:.4f}")

    # ---- DATA SPLIT ----
    n   = len(dataset)
    gen = torch.Generator().manual_seed(42)
    _, _, test_set = random_split(
        dataset,
        [int(0.65*n), int(0.15*n), n - int(0.65*n) - int(0.15*n)],
        generator=gen,
    )
    test_indices = test_set.indices

    # ---- GEO ----
    print("\nLoading geo...")
    lat, lon, full_ext, clip_ext, shp, world_gdf = load_geo()
    lat_is_descending = lat[0] > lat[-1]

    # ---- MODEL ----
    print("\nLoading model...")
    model, embedding_fw = load_model(DEVICE)

    # ---- 2km GT ----
    print("\nLoading 2km GT...")
    gt2km = GT2kmLoader(GT_2KM_FILES, var_name=GT_2KM_VAR)
    gt2km.load()

    # ---- SAMPLE SELECTION ----
    top_samples = find_top_samples_using_gt2km(
        dataset, gt2km, test_indices, time_array,
        lat, lon, PRECIP_MEAN, PRECIP_STD, top_k=10
    )

    print(f"\nRunning inference on {len(top_samples)} samples...")

    for rank, sample_info in enumerate(top_samples):

        dataset_idx = sample_info["dataset_idx"]
        date_str    = sample_info["date_str"]
        tag = f"rank{rank+1}_{date_str}"

        print(f"\n--- {tag} ---")

        # ---- MODEL ----
        lr_s, hr_s = dataset[dataset_idx]
        lr_t = lr_s.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            s2 = model.sample(lr_t, n_steps=N_INFER, embedding_fw=embedding_fw)

        # ---- 8km ----
        gt8_mm = to_mm(hr_s[PRECIP_CH].numpy(), PRECIP_MEAN, PRECIP_STD)
        if lat_is_descending:
            gt8_mm = gt8_mm[::-1, :]
        lat_hr8, lon_hr8 = make_latlon(gt8_mm.shape, lat, lon)

        # ---- 2km GT ----
        gt2_raw, lat2, lon2 = gt2km.get_by_date(date_str)
        if gt2_raw is None:
            continue
        gt2_mm = convert_gt2km_units(gt2_raw)

        # ---- Prediction ----
        pred_mm = to_mm(s2[0, PRECIP_CH].cpu().numpy(), PRECIP_MEAN, PRECIP_STD)
        if lat_is_descending:
            pred_mm = pred_mm[::-1, :]
        lat_pred, lon_pred = make_latlon(pred_mm.shape, lat, lon)

        # ---- CLIP ----
        gt2_sg, lat2_sg, lon2_sg = clip_field(gt2_mm, lat2, lon2, clip_ext)
        pred_sg, lat_ps, lon_ps  = clip_field(pred_mm, lat_pred, lon_pred, clip_ext)

        if gt2_sg.size == 0 or pred_sg.size == 0:
            continue

        # ---- ALIGN ----
        gt2_common, pred_common, lat_c, lon_c = align_to_common_grid(
            gt2_sg, lat2_sg, lon2_sg,
            pred_sg, lat_ps, lon_ps
        )

        # =========================
        # 🔥 BASIC METRICS
        # =========================
        metrics = compute_metrics(gt2_common, pred_common, tag=tag)

        # =========================
        # 🔥 ADVANCED METRICS
        # =========================

        # Blur correlation
        gt_blur   = gaussian_filter(gt2_common, sigma=1)
        pred_blur = gaussian_filter(pred_common, sigma=1)
        r_blur = pearsonr(gt_blur.ravel(), pred_blur.ravel())[0]

        # IoU
        th = 1.0
        gt_mask   = gt2_common > th
        pred_mask = pred_common > th
        iou = np.sum(gt_mask & pred_mask) / (np.sum(gt_mask | pred_mask) + 1e-6)

        # Heavy rain hit rate
        heavy_th = 20
        gt_heavy   = gt2_common > heavy_th
        pred_heavy = pred_common > heavy_th
        hit_rate = np.sum(gt_heavy & pred_heavy) / (np.sum(gt_heavy) + 1e-6)

        # Max error
        gt_max   = float(np.max(gt2_common))
        pred_max = float(np.max(pred_common))
        max_err  = abs(gt_max - pred_max)

        # =========================
        # from scipy.ndimage import gaussian_filter

def main():
    apply_style()
    time_array = load_time()

    print("Loading dataset...")
    dataset = ClimateDataset(LR_FILES, LR_FILES, rank=0)

    global PRECIP_MEAN, PRECIP_STD
    PRECIP_MEAN = dataset.mean["precip"]
    PRECIP_STD  = dataset.std["precip"]
    print(f"  PRECIP_MEAN={PRECIP_MEAN:.4f}  PRECIP_STD={PRECIP_STD:.4f}")

    # ---- DATA SPLIT ----
    n   = len(dataset)
    gen = torch.Generator().manual_seed(42)
    _, _, test_set = random_split(
        dataset,
        [int(0.65*n), int(0.15*n), n - int(0.65*n) - int(0.15*n)],
        generator=gen,
    )
    test_indices = test_set.indices

    # ---- GEO ----
    print("\nLoading geo...")
    lat, lon, full_ext, clip_ext, shp, world_gdf = load_geo()
    lat_is_descending = lat[0] > lat[-1]

    # ---- MODEL ----
    print("\nLoading model...")
    model, embedding_fw = load_model(DEVICE)

    # ---- 2km GT ----
    print("\nLoading 2km GT...")
    gt2km = GT2kmLoader(GT_2KM_FILES, var_name=GT_2KM_VAR)
    gt2km.load()

    # ---- SAMPLE SELECTION ----
    top_samples = find_top_samples_using_gt2km(
        dataset, gt2km, test_indices, time_array,
        lat, lon, PRECIP_MEAN, PRECIP_STD, top_k=10
    )

    print(f"\nRunning inference on {len(top_samples)} samples...")

    for rank, sample_info in enumerate(top_samples):

        dataset_idx = sample_info["dataset_idx"]
        date_str    = sample_info["date_str"]
        tag = f"rank{rank+1}_{date_str}"

        print(f"\n--- {tag} ---")

        # ---- MODEL ----
        lr_s, hr_s = dataset[dataset_idx]
        lr_t = lr_s.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            s2 = model.sample(lr_t, n_steps=N_INFER, embedding_fw=embedding_fw)

        # ---- 8km ----
        gt8_mm = to_mm(hr_s[PRECIP_CH].numpy(), PRECIP_MEAN, PRECIP_STD)
        if lat_is_descending:
            gt8_mm = gt8_mm[::-1, :]
        lat_hr8, lon_hr8 = make_latlon(gt8_mm.shape, lat, lon)

        # ---- 2km GT ----
        gt2_raw, lat2, lon2 = gt2km.get_by_date(date_str)
        if gt2_raw is None:
            continue
        gt2_mm = convert_gt2km_units(gt2_raw)

        # ---- Prediction ----
        pred_mm = to_mm(s2[0, PRECIP_CH].cpu().numpy(), PRECIP_MEAN, PRECIP_STD)
        if lat_is_descending:
            pred_mm = pred_mm[::-1, :]
        lat_pred, lon_pred = make_latlon(pred_mm.shape, lat, lon)

        # ---- CLIP ----
        gt2_sg, lat2_sg, lon2_sg = clip_field(gt2_mm, lat2, lon2, clip_ext)
        pred_sg, lat_ps, lon_ps  = clip_field(pred_mm, lat_pred, lon_pred, clip_ext)

        if gt2_sg.size == 0 or pred_sg.size == 0:
            continue

        # ---- ALIGN ----
        gt2_common, pred_common, lat_c, lon_c = align_to_common_grid(
            gt2_sg, lat2_sg, lon2_sg,
            pred_sg, lat_ps, lon_ps
        )

        # =========================
        # 🔥 BASIC METRICS
        # =========================
        metrics = compute_metrics(gt2_common, pred_common, tag=tag)

        # =========================
        # 🔥 ADVANCED METRICS
        # =========================

        # Blur correlation
        gt_blur   = gaussian_filter(gt2_common, sigma=1)
        pred_blur = gaussian_filter(pred_common, sigma=1)
        r_blur = pearsonr(gt_blur.ravel(), pred_blur.ravel())[0]

        # IoU
        th = 1.0
        gt_mask   = gt2_common > th
        pred_mask = pred_common > th
        iou = np.sum(gt_mask & pred_mask) / (np.sum(gt_mask | pred_mask) + 1e-6)

        # Heavy rain hit rate
        heavy_th = 20
        gt_heavy   = gt2_common > heavy_th
        pred_heavy = pred_common > heavy_th
        hit_rate = np.sum(gt_heavy & pred_heavy) / (np.sum(gt_heavy) + 1e-6)

        # Max error
        gt_max   = float(np.max(gt2_common))
        pred_max = float(np.max(pred_common))
        max_err  = abs(gt_max - pred_max)

        # =========================
        # 🔥 PRINT
        # =========================
        print(f"""
        RMSE        : {metrics['rmse']:.3f}
        MAE         : {metrics['mae']:.3f}
        Bias        : {metrics['bias']:.3f}
        PCC         : {metrics['pearson_r']:.3f}

        Blur PCC    : {r_blur:.3f}
        IoU (>1mm)  : {iou:.3f}
        HitRate>20  : {hit_rate:.3f}
        Max Error   : {max_err:.3f}
        """)

        # FIX 2: In main loop, after compute_metrics block, ADD THESE CALLS:

        # ---- 8km clipped (needed for plot) ----
        gt8_sg, lat8_sg, lon8_sg = clip_field(gt8_mm, lat_hr8, lon_hr8, clip_ext)

        sample_dir = os.path.join(OUTPUT_DIR, tag)
        os.makedirs(sample_dir, exist_ok=True)

        # ---- THREE-PANEL PLOT ----
        fig = build_three_panel(
            gt8_sg      = gt8_sg,
            gt2_common  = gt2_common,
            pred_common = pred_common,
            lat_gt2     = lat_c,
            lon_gt2     = lon_c,
            shp         = shp,
            world_gdf   = world_gdf,
            metrics_gt2_vs_pred = metrics,
            title_tag   = f"{tag}  |  2km-max={sample_info['max_2km_gt']:.1f} "
        )
        _save(fig, os.path.join(sample_dir, "three_panel.png"))

        # ---- SCATTER ----
        plot_scatter(gt2_common, pred_common, metrics, tag, sample_dir)

        # ---- HISTOGRAMS ----
        plot_rainfall_histograms(gt2_common, pred_common, tag, sample_dir)

        # ---- CSV ----
        metrics.update({"r_blur": r_blur, "iou": iou,
                        "hit_rate": hit_rate, "max_error": max_err})
        pd.DataFrame([metrics]).to_csv(
            os.path.join(sample_dir, "metrics_full.csv"), index=False)
    print("\n✅ DONE")
    

    print("\n✅ DONE")

if __name__ == "__main__":
    main()
  
