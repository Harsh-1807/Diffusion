"""
Singapore domain: GT (June 1995, native 2km) vs CorrDiff Diffusion-Mean output,
restricted to June 1995 only.

GT source:  pr_V3-WMC-2_ERA5_historical_reanalysis_SINGV-RCM_vn5_day_19950601-19950630.nc
            variable 'pr', units kg m-2 s-1 -> converted to mm/day via x*86400
            grid: 960 (lat) x 960 (lon)

Comparison: precip_2km_1995.nc
            variables 'precip_mean' (ensemble mean, mm/day) and 'precip_std'
            (ensemble std, mm/day), full year (365 days) -> sliced to June
            grid: 1024 (lat) x 1088 (lon)

Because the two products are on DIFFERENT 2km grids (different extent/shape),
this script regrids the Diffusion product onto the GT's native lat/lon grid
via linear interpolation (xarray .interp), so every downstream comparison
(maps, histograms, scatter, metrics) is pixel-paired against GT.

WHY THESE PARTICULAR METRICS:
Point-wise metrics (RMSE, bias, R2, pixel scatter) structurally punish a
stochastic diffusion field for small spatial displacement (the "double
penalty" problem), and time-averaging erases the fine-scale texture
diffusion is meant to add before any metric even runs. The set below is
the standard toolkit for evaluating a probabilistic downscaling product
fairly:
  - FSS (Fractions Skill Score): neighborhood-tolerant, fixes double penalty.
  - Per-DAY PSD (not PSD-of-time-mean): shows whether fine-scale spectral
    power survives on individual days.
  - POD / FAR / CSI at rain-rate thresholds: standard contingency skill
    scores, often favorable to diffusion even when RMSE isn't.
  - Spread-skill correlation: is the ensemble's predicted uncertainty (std)
    actually tracking where the model is wrong?
  - CRPS (Gaussian-approximated from mean+std): the proper scoring rule for
    a probabilistic forecast, using the closed-form Gaussian CRPS since we
    only have mean/std rather than full ensemble members.
  - Taylor diagram: summarizes pattern correlation, normalized spatial
    std-dev, and centered RMSE for the June-mean field in one plot.
  - RMSE decomposition: total RMSE split into bias^2 and unbiased
    (centered) RMSE, since a small total RMSE can hide a big systematic bias.
  - Extreme-day (June max) comparison: does the model capture the single
    heaviest rain day per pixel, which the mean-field metrics wash out.

All figures / metrics are written into ./Results_June1995 (created automatically).

Figures produced:
  1.  precip_comparison_maps_june1995.png    - GT / Diffusion-mean / Std / diff map
  2.  precip_histogram_june1995.png          - value distribution, line style
  3.  precip_daily_timeseries_june1995.png   - daily spatial-mean time series (30 days)
  4.  precip_psd_timemean_june1995.png       - PSD of the TIME-MEAN field (reference view)
  5.  precip_psd_perday_avg_june1995.png     - PSD averaged across each day (fairer to diffusion)
  6.  precip_qq_june1995.png                 - quantile-quantile plot vs GT
  7.  precip_scatter_vs_gt_june1995.png      - pixel-wise scatter (regridded vs GT, June-mean field)
  8.  precip_cdf_june1995.png                - empirical CDF comparison
  9.  precip_fss_june1995.png                - Fractions Skill Score vs neighborhood size
 10.  precip_contingency_june1995.png        - POD / FAR / CSI bars at rain-rate thresholds
 11.  precip_spread_skill_june1995.png       - ensemble std vs |error|, scatter + correlation
 12.  precip_crps_map_june1995.png           - spatial map of Gaussian-approx CRPS
 13.  precip_taylor_diagram_june1995.png     - Taylor diagram (pattern corr / std / centered RMSE)
 14.  precip_rmse_decomposition_june1995.png - RMSE split into bias^2 vs unbiased RMSE
 15.  precip_extreme_day_june1995.png        - June-max-per-pixel comparison map
 16.  precip_metrics_bar_june1995.png        - summary metric comparison bars

Also writes (into Results_June1995/): metrics_summary_june1995.csv / .txt,
fss_summary_june1995.csv, contingency_summary_june1995.csv
"""

import os
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import math
import csv
import warnings

warnings.filterwarnings("ignore")

try:
    from scipy.ndimage import uniform_filter
    _HAVE_SCIPY_NDIMAGE = True
except Exception:
    _HAVE_SCIPY_NDIMAGE = False

try:
    from scipy.stats import norm as _norm
    _HAVE_SCIPY_STATS = True
except Exception:
    _HAVE_SCIPY_STATS = False

# ------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------
FILE_PATH_GT = (
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Singapore_Data/GT_from_Prasanna/1995/"
    "pr_V3-WMC-2_ERA5_historical_reanalysis_SINGV-RCM_vn5_day_19950601-19950630.nc"
)
FILE_PATH_COMPARE = (
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/VERSION4/outputs/"
    "downscaled_2km/precip_2km_1995.nc"
)

GT_VAR = "pr"                       # kg m-2 s-1 -> mm/day via *86400
GT_TO_MMDAY = 86400.0

COMPARE_VAR_DIFF = "precip_mean"    # already mm/day (diffusion ensemble mean)
COMPARE_VAR_STD = "precip_std"      # already mm/day (diffusion ensemble std)

# FSS neighborhood window sizes, in GRID CELLS (GT native grid = 2km/pixel)
FSS_WINDOWS_PX = [1, 3, 5, 9, 15, 21, 31]   # -> 2, 6, 10, 18, 30, 42, 62 km
FSS_THRESHOLDS_MMDAY = [1.0, 10.0, 25.0]    # light / moderate / heavy rain

# Contingency-table thresholds (pooled over all June days + pixels)
CONTINGENCY_THRESHOLDS_MMDAY = [1.0, 10.0, 25.0, 50.0]

RESULTS_DIR = "Results_June1995"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Visual style: consistent two-color palette + cleaner global look
# ------------------------------------------------------------------
COLOR_GT = "#2C3E50"       # dark slate blue
COLOR_DIFF = "#E67E22"     # warm orange
COLOR_ACCENT = "#7F8C8D"   # neutral gray for reference lines

STYLE = {
    "GT":             dict(color=COLOR_GT, linestyle="-", marker="o", lw=2.4, ms=4),
    "Diffusion Mean": dict(color=COLOR_DIFF, linestyle="-", marker="^", lw=2.4, ms=4),
}

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "medium",
    "axes.edgecolor": "#4A4A4A",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#B0B0B0",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.fontsize": 11,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})


def out_path(filename):
    return os.path.join(RESULTS_DIR, filename)


# ------------------------------------------------------------------
# 2. Colormap helper
# ------------------------------------------------------------------
def make_dynamic_cmap(vmax, n_levels=9, kind="rain", vmin=None):
    vmax = max(float(vmax), 1e-6)

    if kind == "rain":
        raw_step = vmax / max(n_levels - 1, 1)
        magnitude = 10 ** math.floor(math.log10(max(raw_step, 1e-9)))
        step = magnitude * round(raw_step / magnitude)
        if step <= 0:
            step = raw_step

        lo = 0.0 if vmin is None else float(vmin)
        bounds = np.arange(lo, vmax + step, step)
        if bounds[-1] < vmax:
            bounds = np.append(bounds, bounds[-1] + step)
        bounds = np.round(bounds, 6)

        cmap = plt.get_cmap("viridis", len(bounds) - 1)
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


# ------------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------------
def clean_flat(arr):
    a = np.asarray(arr, dtype=float).flatten()
    return a[np.isfinite(a)]


def radial_psd(field2d, dx):
    field = np.asarray(field2d, dtype=float)
    fill_val = np.nanmean(field)
    field = np.where(np.isnan(field), fill_val, field)
    field = field - field.mean()

    ny, nx = field.shape
    fft2 = np.fft.fftshift(np.fft.fft2(field))
    psd2d = (np.abs(fft2) ** 2) / (nx * ny)

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dx))
    kx2d, ky2d = np.meshgrid(kx, ky)
    k_r = np.sqrt(kx2d ** 2 + ky2d ** 2)

    n_bins = max(min(nx, ny) // 2, 4)
    k_edges = np.linspace(0, k_r.max(), n_bins + 1)
    k_mid = 0.5 * (k_edges[1:] + k_edges[:-1])
    psd_r = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (k_r >= k_edges[i]) & (k_r < k_edges[i + 1])
        if mask.any():
            psd_r[i] = psd2d[mask].mean()

    keep = k_mid > 0
    return k_mid[keep], psd_r[keep]


def radial_psd_perday_avg(field3d, dx, time_axis=0):
    field3d = np.asarray(field3d, dtype=float)
    n_t = field3d.shape[time_axis]
    psd_stack = None
    k_ref = None
    for t in range(n_t):
        field2d = np.take(field3d, t, axis=time_axis)
        k, psd = radial_psd(field2d, dx)
        if psd_stack is None:
            psd_stack = np.full((n_t, len(psd)), np.nan)
            k_ref = k
        psd_stack[t, :] = psd
    return k_ref, np.nanmean(psd_stack, axis=0)


def quantile_pair(ref, comp, probs):
    ref_v, comp_v = clean_flat(ref), clean_flat(comp)
    return np.percentile(ref_v, probs), np.percentile(comp_v, probs)


def wet_day_fraction(daily_domain_mean, thresh=1.0):
    vals = clean_flat(daily_domain_mean)
    return 100.0 * np.mean(vals > thresh) if vals.size else np.nan


def paired_error_stats(ref2d, comp2d):
    a = np.asarray(ref2d, dtype=float).flatten()
    b = np.asarray(comp2d, dtype=float).flatten()
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    bias = float(np.mean(b - a))
    rmse = float(np.sqrt(np.mean((b - a) ** 2)))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else np.nan
    r2 = corr ** 2 if np.isfinite(corr) else np.nan
    # RMSE decomposition: total RMSE^2 = bias^2 + unbiased(centered) RMSE^2
    unbiased_rmse = float(np.sqrt(max(rmse ** 2 - bias ** 2, 0.0)))
    return dict(bias=bias, rmse=rmse, corr=corr, r2=r2, n=int(a.size),
                unbiased_rmse=unbiased_rmse), a, b


def _uniform_filter_fallback(arr, size):
    pad = size // 2
    arr_p = np.pad(arr, pad, mode="constant", constant_values=0.0)
    csum = np.cumsum(np.cumsum(arr_p, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    ny, nx = arr.shape
    i_idx = np.arange(ny)
    j_idx = np.arange(nx)
    i0, i1 = i_idx, i_idx + size
    j0, j1 = j_idx, j_idx + size
    total = (csum[np.ix_(i1, j1)] - csum[np.ix_(i0, j1)]
             - csum[np.ix_(i1, j0)] + csum[np.ix_(i0, j0)])
    return total / float(size * size)


def box_filter(arr, size):
    if size <= 1:
        return arr.astype(float)
    if _HAVE_SCIPY_NDIMAGE:
        return uniform_filter(arr.astype(float), size=size, mode="constant", cval=0.0)
    return _uniform_filter_fallback(arr.astype(float), size)


def fss_single_day(ref2d, fcst2d, threshold, window):
    ref_bin = (ref2d > threshold).astype(float)
    fcst_bin = (fcst2d > threshold).astype(float)
    ref_frac = box_filter(ref_bin, window)
    fcst_frac = box_filter(fcst_bin, window)
    mse = np.nanmean((ref_frac - fcst_frac) ** 2)
    mse_ref = np.nanmean(ref_frac ** 2) + np.nanmean(fcst_frac ** 2)
    if mse_ref <= 0:
        return np.nan
    return 1.0 - (mse / mse_ref)


def fss_over_days(ref3d, fcst3d, threshold, window, time_axis=0):
    n_t = ref3d.shape[time_axis]
    vals = []
    for t in range(n_t):
        r2d = np.take(ref3d, t, axis=time_axis)
        f2d = np.take(fcst3d, t, axis=time_axis)
        v = fss_single_day(r2d, f2d, threshold, window)
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def contingency_scores(ref_flat, comp_flat, thresh):
    ref_flat = np.asarray(ref_flat, dtype=float)
    comp_flat = np.asarray(comp_flat, dtype=float)
    valid = np.isfinite(ref_flat) & np.isfinite(comp_flat)
    r, c = ref_flat[valid], comp_flat[valid]

    hits = np.sum((r >= thresh) & (c >= thresh))
    misses = np.sum((r >= thresh) & (c < thresh))
    false_alarms = np.sum((r < thresh) & (c >= thresh))

    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan
    return dict(pod=float(pod), far=float(far), csi=float(csi),
                hits=int(hits), misses=int(misses), false_alarms=int(false_alarms))


def _normal_cdf(z):
    if _HAVE_SCIPY_STATS:
        return _norm.cdf(z)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))


def _normal_pdf(z):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2)


def crps_gaussian(obs, mu, sigma):
    """
    Closed-form CRPS for a Gaussian forecast N(mu, sigma^2) against a
    deterministic observation `obs` (Gneiting et al. 2005). Used here as an
    approximation since only the ensemble mean/std are available, not the
    full ensemble.
    """
    sigma_safe = np.where(sigma <= 1e-6, 1e-6, sigma)
    z = (obs - mu) / sigma_safe
    phi = _normal_pdf(z)
    Phi = _normal_cdf(z)
    crps = sigma_safe * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
    return crps


def select_june_1995(ds, time_dim):
    try:
        out = ds.sel({time_dim: slice("1995-06-01", "1995-06-30")})
        if len(out[time_dim]) == 0:
            raise ValueError("empty selection")
        return out
    except Exception:
        return ds.isel({time_dim: slice(151, 181)})


def main():
    print("Loading GT (June 1995)...")
    ds_gt = xr.open_dataset(FILE_PATH_GT)
    time_dim_gt = [d for d in ds_gt.dims if "time" in d.lower()][0]
    ds_gt_june = select_june_1995(ds_gt, time_dim_gt)

    gt_mmday = ds_gt_june[GT_VAR] * GT_TO_MMDAY
    gt_mmday.attrs["units"] = "mm/day"

    print("Loading Diffusion-mean product and restricting to June 1995...")
    ds_cmp = xr.open_dataset(FILE_PATH_COMPARE)
    time_dim_cmp = [d for d in ds_cmp.dims if "time" in d.lower()][0]
    ds_cmp_june = select_june_1995(ds_cmp, time_dim_cmp)

    n_gt, n_cmp = len(gt_mmday[time_dim_gt]), len(ds_cmp_june[time_dim_cmp])
    if n_gt != n_cmp:
        print(f"WARNING: GT has {n_gt} June timesteps, comparison has {n_cmp}. "
              f"Trimming both to the shorter length.")
        n_min = min(n_gt, n_cmp)
        gt_mmday = gt_mmday.isel({time_dim_gt: slice(0, n_min)})
        ds_cmp_june = ds_cmp_june.isel({time_dim_cmp: slice(0, n_min)})

    diff_june = ds_cmp_june[COMPARE_VAR_DIFF]
    std_june = ds_cmp_june[COMPARE_VAR_STD]

    print("Regridding Diffusion product onto GT grid (linear interp)...")
    gt_lat = ds_gt_june["lat"]
    gt_lon = ds_gt_june["lon"]

    def regrid_to_gt(da):
        return da.interp(lat=gt_lat, lon=gt_lon, method="linear")

    diff_on_gt = regrid_to_gt(diff_june)
    std_on_gt = regrid_to_gt(std_june)

    gt_arr = gt_mmday.transpose(time_dim_gt, "lat", "lon").values
    diff_arr = diff_on_gt.transpose(time_dim_cmp, "lat", "lon").values
    std_arr = std_on_gt.transpose(time_dim_cmp, "lat", "lon").values

    mean_gt = gt_mmday.mean(dim=time_dim_gt)
    mean_diff = diff_on_gt.mean(dim=time_dim_cmp)
    mean_std = std_on_gt.mean(dim=time_dim_cmp)

    stats_diff, a_diff, b_diff = paired_error_stats(mean_gt.values, mean_diff.values)

    # ==========================================
    # FIGURE 1: Spatial maps (GT / Diffusion-mean / Std / diff map)
    # ==========================================
    print("Generating spatial comparison maps...")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), constrained_layout=True)

    max_val = float(max(mean_gt.max().values, mean_diff.max().values))
    cmap_r, norm_r, _ = make_dynamic_cmap(vmax=max_val, n_levels=10, kind="rain")

    im = axes[0].imshow(mean_gt.values, cmap=cmap_r, norm=norm_r, origin="lower")
    axes[0].set_title("GT (June 1995 mean)")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], orientation="horizontal", pad=0.06, fraction=0.05, label="mm/day")

    im = axes[1].imshow(mean_diff.values, cmap=cmap_r, norm=norm_r, origin="lower")
    axes[1].set_title("Diffusion Mean (June 1995 mean)")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], orientation="horizontal", pad=0.06, fraction=0.05, label="mm/day")

    std_max = float(np.nanmax(mean_std.values))
    im = axes[2].imshow(mean_std.values, cmap="magma", vmin=0, vmax=max(std_max, 1e-3), origin="lower")
    axes[2].set_title("Ensemble Uncertainty (Std)")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.colorbar(im, ax=axes[2], orientation="horizontal", pad=0.06, fraction=0.05, label="mm/day")

    diff_field = mean_diff.values - mean_gt.values
    diff_max = float(np.nanmax(np.abs(diff_field)))
    cmap_e, norm_e, _ = make_dynamic_cmap(vmax=diff_max, n_levels=8, kind="diff")
    im = axes[3].imshow(diff_field, cmap=cmap_e, norm=norm_e, origin="lower")
    axes[3].set_title(f"Diffusion \u2212 GT\n(Bias={stats_diff['bias']:.2f}, RMSE={stats_diff['rmse']:.2f} mm/day)")
    axes[3].set_xticks([]); axes[3].set_yticks([])
    fig.colorbar(im, ax=axes[3], orientation="horizontal", pad=0.06, fraction=0.05, label="mm/day diff")

    fig.suptitle("GT vs Diffusion-Mean: June 1995, Regridded to GT's Native 2km Grid")
    plt.savefig(out_path("precip_comparison_maps_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 2: Value distribution (line histogram)
    # ==========================================
    print("Generating distribution line plot...")
    flat_gt = clean_flat(gt_mmday.values)
    flat_diff = clean_flat(diff_on_gt.values)

    all_vals = np.concatenate([flat_gt, flat_diff])
    bin_edges = np.linspace(0, np.percentile(all_vals, 99.9), 51)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    plt.figure(figsize=(9, 5.5))
    for label, data in [("GT", flat_gt), ("Diffusion Mean", flat_diff)]:
        counts, _ = np.histogram(data, bins=bin_edges, density=True)
        plt.plot(bin_centers, counts, label=label, **STYLE[label])
        plt.fill_between(bin_centers, counts, alpha=0.08, color=STYLE[label]["color"])

    plt.yscale("log")
    plt.title("Precipitation Value Distribution \u2014 June 1995")
    plt.xlabel("Precipitation Intensity (mm/day)")
    plt.ylabel("Density (log scale)")
    plt.legend()
    plt.savefig(out_path("precip_histogram_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 3: Daily spatial-mean time series
    # ==========================================
    print("Generating daily spatial-mean time series...")
    spatial_mean_gt = gt_mmday.mean(dim=[d for d in gt_mmday.dims if d != time_dim_gt]).values
    spatial_mean_diff = diff_on_gt.mean(dim=[d for d in diff_on_gt.dims if d != time_dim_cmp]).values

    plt.figure(figsize=(11, 5))
    days_axis = np.arange(1, len(spatial_mean_gt) + 1)
    for label, data in [("GT", spatial_mean_gt), ("Diffusion Mean", spatial_mean_diff)]:
        plt.plot(days_axis, data, label=label, **STYLE[label])

    plt.title("June 1995: Daily Spatial-Mean Precipitation")
    plt.xlabel("Day of June")
    plt.ylabel("Spatial Mean Precipitation (mm/day)")
    plt.xticks(days_axis[::2])
    plt.legend()
    plt.savefig(out_path("precip_daily_timeseries_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 4: PSD of the TIME-MEAN field
    # ==========================================
    print("Generating time-mean PSD plot...")
    DX_GT_KM = 2.0
    k_gt, psd_gt = radial_psd(mean_gt.values, dx=DX_GT_KM)
    k_diff, psd_diff = radial_psd(mean_diff.values, dx=DX_GT_KM)

    plt.figure(figsize=(8.5, 5.5))
    for label, (k, psd) in [("GT", (k_gt, psd_gt)), ("Diffusion Mean", (k_diff, psd_diff))]:
        plt.loglog(k, psd, label=label, color=STYLE[label]["color"], linewidth=2.4)

    plt.title("PSD of TIME-MEAN Field \u2014 June 1995\n(time-averaging washes out fine-scale texture)",
              fontsize=12)
    plt.xlabel("Wavenumber (cycles / km)")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.savefig(out_path("precip_psd_timemean_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 5: PSD averaged PER-DAY
    # ==========================================
    print("Generating per-day-averaged PSD plot...")
    k_gt_pd, psd_gt_pd = radial_psd_perday_avg(gt_arr, dx=DX_GT_KM)
    k_df_pd, psd_df_pd = radial_psd_perday_avg(diff_arr, dx=DX_GT_KM)

    plt.figure(figsize=(8.5, 5.5))
    for label, (k, psd) in [("GT", (k_gt_pd, psd_gt_pd)), ("Diffusion Mean", (k_df_pd, psd_df_pd))]:
        plt.loglog(k, psd, label=label, color=STYLE[label]["color"], linewidth=2.4)

    plt.title("PSD Averaged Across Each Day \u2014 June 1995\n(fairer to diffusion's daily texture)",
              fontsize=12)
    plt.xlabel("Wavenumber (cycles / km)")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.savefig(out_path("precip_psd_perday_avg_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 6: QQ plot vs GT
    # ==========================================
    print("Generating QQ plot...")
    probs = np.concatenate([np.linspace(1, 95, 60), np.linspace(95, 99.9, 30)])
    ref_q, diff_q = quantile_pair(flat_gt, flat_diff, probs)
    lims = [0, max(ref_q.max(), diff_q.max()) * 1.05]

    plt.figure(figsize=(6.8, 6.8))
    plt.plot(ref_q, diff_q, color=STYLE["Diffusion Mean"]["color"], marker="^",
              linestyle="none", ms=6, alpha=0.85, label="Diffusion vs GT")
    plt.plot(lims, lims, color=COLOR_ACCENT, linestyle=":", linewidth=1.6, label="1:1 line")

    plt.xlim(lims); plt.ylim(lims)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Quantile-Quantile Plot vs GT \u2014 June 1995")
    plt.xlabel("GT Quantiles (mm/day)")
    plt.ylabel("Diffusion Mean Quantiles (mm/day)")
    plt.legend()
    plt.savefig(out_path("precip_qq_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 7: Pixel-wise scatter
    # ==========================================
    print("Generating scatter plot...")
    plt.figure(figsize=(7, 6.5))
    sc_lim = [0, max(a_diff.max(), b_diff.max()) * 1.05]
    plt.scatter(a_diff, b_diff, s=10, alpha=0.3, color=STYLE["Diffusion Mean"]["color"],
                edgecolors="none")
    plt.plot(sc_lim, sc_lim, color=COLOR_ACCENT, linestyle=":", linewidth=1.6, label="1:1 line")
    plt.title(f"Diffusion Mean vs GT \u2014 June 1995\n"
              f"R\u00b2={stats_diff['r2']:.3f}  |  RMSE={stats_diff['rmse']:.2f} mm/day  |  "
              f"Bias={stats_diff['bias']:.2f} mm/day")
    plt.xlabel("GT (mm/day)")
    plt.ylabel("Diffusion Mean (mm/day)")
    plt.xlim(sc_lim); plt.ylim(sc_lim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.savefig(out_path("precip_scatter_vs_gt_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 8: Empirical CDF
    # ==========================================
    print("Generating CDF comparison plot...")
    plt.figure(figsize=(8.5, 5.5))
    for label, data in [("GT", flat_gt), ("Diffusion Mean", flat_diff)]:
        sorted_vals = np.sort(data)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        plt.plot(sorted_vals, cdf, label=label, color=STYLE[label]["color"], linewidth=2.4)

    plt.xlim(0, np.percentile(all_vals, 99.5))
    plt.title("Empirical CDF \u2014 June 1995")
    plt.xlabel("Precipitation Intensity (mm/day)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.savefig(out_path("precip_cdf_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 9: Fractions Skill Score (FSS)
    # ==========================================
    print("Computing FSS (loops over days x windows x thresholds)...")
    if not _HAVE_SCIPY_NDIMAGE:
        print("  NOTE: scipy not found, using slower pure-numpy box filter fallback.")

    fss_results = {}
    for thresh in FSS_THRESHOLDS_MMDAY:
        fss_results[thresh] = [fss_over_days(gt_arr, diff_arr, thresh, w) for w in FSS_WINDOWS_PX]

    fig, axes = plt.subplots(1, len(FSS_THRESHOLDS_MMDAY), figsize=(6 * len(FSS_THRESHOLDS_MMDAY), 5),
                              constrained_layout=True, sharey=True)
    if len(FSS_THRESHOLDS_MMDAY) == 1:
        axes = [axes]
    window_km = [w * DX_GT_KM for w in FSS_WINDOWS_PX]
    for ax, thresh in zip(axes, FSS_THRESHOLDS_MMDAY):
        ax.plot(window_km, fss_results[thresh], color=STYLE["Diffusion Mean"]["color"],
                 marker="^", linewidth=2.4, ms=7, label="Diffusion Mean")
        ax.axhline(0.5, color=COLOR_ACCENT, linestyle=":", linewidth=1.4, label="Useful-skill (0.5)")
        ax.set_title(f"Threshold > {thresh:g} mm/day")
        ax.set_xlabel("Neighborhood size (km)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Fractions Skill Score")
    axes[0].legend()
    fig.suptitle("FSS vs Neighborhood Size, Averaged Over June 1995 Days")
    plt.savefig(out_path("precip_fss_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 10: Contingency-table skill scores
    # ==========================================
    print("Computing pooled contingency-table scores (POD/FAR/CSI)...")
    gt_flat_all = gt_arr.flatten()
    diff_flat_all = diff_arr.flatten()

    contingency_rows = []
    for thresh in CONTINGENCY_THRESHOLDS_MMDAY:
        cdf_ = contingency_scores(gt_flat_all, diff_flat_all, thresh)
        contingency_rows.append((thresh, cdf_))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    metric_names = ["pod", "far", "csi"]
    metric_titles = ["POD (higher is better)", "FAR (lower is better)", "CSI (higher is better)"]
    x = np.arange(len(CONTINGENCY_THRESHOLDS_MMDAY))

    for ax, mname, mtitle in zip(axes, metric_names, metric_titles):
        vals = [row[1][mname] for row in contingency_rows]
        bars = ax.bar(x, vals, width=0.55, color=STYLE["Diffusion Mean"]["color"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f">{t:g}" for t in CONTINGENCY_THRESHOLDS_MMDAY])
        ax.set_xlabel("Rain rate threshold (mm/day)")
        ax.set_title(mtitle)
        ax.set_ylim(0, 1.08)

    fig.suptitle("Contingency-Table Skill Scores \u2014 Diffusion Mean vs GT, Pooled Over June 1995")
    plt.savefig(out_path("precip_contingency_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 11: Spread-skill relationship
    # ==========================================
    print("Generating spread-skill diagnostic...")
    abs_err = np.abs(mean_diff.values - mean_gt.values)
    std_v = mean_std.values.flatten()
    err_v = abs_err.flatten()
    valid_ss = np.isfinite(std_v) & np.isfinite(err_v)
    std_v, err_v = std_v[valid_ss], err_v[valid_ss]
    spread_skill_corr = float(np.corrcoef(std_v, err_v)[0, 1]) if std_v.size > 1 else np.nan

    plt.figure(figsize=(7, 6))
    plt.scatter(std_v, err_v, s=8, alpha=0.25, color=STYLE["Diffusion Mean"]["color"], edgecolors="none")
    plt.title(f"Spread-Skill \u2014 Ensemble Std vs |Error| (June-mean field)\n"
              f"Correlation = {spread_skill_corr:.3f}")
    plt.xlabel("Ensemble Std (mm/day)")
    plt.ylabel("|Diffusion Mean \u2212 GT| (mm/day)")
    plt.savefig(out_path("precip_spread_skill_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 12: CRPS map (Gaussian approximation)
    # ==========================================
    print("Computing Gaussian-approximated CRPS...")
    crps_field = crps_gaussian(mean_gt.values, mean_diff.values, mean_std.values)
    crps_mean = float(np.nanmean(crps_field))

    plt.figure(figsize=(7.5, 6.5))
    crps_max = float(np.nanpercentile(np.abs(crps_field), 99))
    im = plt.imshow(crps_field, cmap="inferno", vmin=0, vmax=max(crps_max, 1e-3), origin="lower")
    plt.title(f"Gaussian-Approx. CRPS \u2014 June-Mean Field\nDomain-mean CRPS = {crps_mean:.3f} mm/day "
              f"(lower is better)")
    plt.xticks([]); plt.yticks([])
    plt.colorbar(im, orientation="vertical", fraction=0.046, pad=0.04, label="CRPS (mm/day)")
    plt.savefig(out_path("precip_crps_map_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 13: Taylor diagram
    # ==========================================
    print("Generating Taylor diagram...")
    std_gt_spatial = float(np.nanstd(mean_gt.values))
    std_diff_spatial = float(np.nanstd(mean_diff.values))
    norm_std = std_diff_spatial / std_gt_spatial if std_gt_spatial > 0 else np.nan
    pattern_corr = stats_diff["corr"]
    theta = np.arccos(np.clip(pattern_corr, -1, 1))

    max_radius = max(1.3, norm_std * 1.2)

    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    corr_ticks = [1, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0]
    ax.set_xticks(np.arccos(corr_ticks))
    ax.set_xticklabels([f"{c:g}" for c in corr_ticks])
    ax.set_ylim(0, max_radius)
    ax.set_rlabel_position(135)

    theta_grid = np.linspace(0, np.pi / 2, 150)
    r_grid = np.linspace(0.001, max_radius, 150)
    T, R = np.meshgrid(theta_grid, r_grid)
    RMSD = np.sqrt(1 + R ** 2 - 2 * R * np.cos(T))
    cs = ax.contour(T, R, RMSD, levels=[0.25, 0.5, 0.75, 1.0, 1.25], colors=COLOR_ACCENT,
                     linestyles=":", linewidths=0.9)
    ax.clabel(cs, inline=True, fontsize=8.5, fmt="%.2f")

    ax.plot(0, 1, marker="o", color=COLOR_GT, markersize=13, linestyle="none",
             label="GT (reference)", zorder=5)
    ax.plot(theta, norm_std, marker="^", color=COLOR_DIFF, markersize=13, linestyle="none",
             label="Diffusion Mean", zorder=5)

    ax.set_title("Taylor Diagram \u2014 June-Mean Field Pattern Skill\n"
                 "(radius = normalized spatial std-dev, angle = pattern correlation,\n"
                 "dotted contours = centered RMSE, normalized)", fontsize=11.5, pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.savefig(out_path("precip_taylor_diagram_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 14: RMSE decomposition (bias^2 vs unbiased RMSE)
    # ==========================================
    print("Generating RMSE decomposition chart...")
    plt.figure(figsize=(6.5, 5.5))
    components = ["Bias", "Unbiased (centered) RMSE", "Total RMSE"]
    values = [abs(stats_diff["bias"]), stats_diff["unbiased_rmse"], stats_diff["rmse"]]
    bar_colors = ["#95A5A6", STYLE["Diffusion Mean"]["color"], COLOR_GT]
    bars = plt.bar(components, values, color=bar_colors, width=0.55)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.015, f"{v:.2f}",
                  ha="center", va="bottom", fontsize=10.5)
    plt.ylabel("mm/day")
    plt.title("RMSE Decomposition \u2014 Diffusion Mean vs GT (June-mean field)\n"
              "Total RMSE\u00b2 = Bias\u00b2 + Unbiased RMSE\u00b2", fontsize=12)
    plt.xticks(rotation=10)
    plt.savefig(out_path("precip_rmse_decomposition_june1995.png"))
    plt.close()

    # ==========================================
    # FIGURE 15: Extreme-day (June max per pixel) comparison
    # ==========================================
    print("Generating extreme-day (June max) comparison map...")
    max_gt = np.nanmax(gt_arr, axis=0)
    max_diff = np.nanmax(diff_arr, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    max_val_extreme = float(max(np.nanmax(max_gt), np.nanmax(max_diff)))
    cmap_x, norm_x, _ = make_dynamic_cmap(vmax=max_val_extreme, n_levels=10, kind="rain")

    im = axes[0].imshow(max_gt, cmap=cmap_x, norm=norm_x, origin="lower")
    axes[0].set_title("GT \u2014 June 1995 Max (per pixel)")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], orientation="horizontal", pad=0.06, fraction=0.05, label="mm/day")

    im = axes[1].imshow(max_diff, cmap=cmap_x, norm=norm_x, origin="lower")
    axes[1].set_title("Diffusion Mean \u2014 June 1995 Max (per pixel)")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.colorbar(im, ax=axes[1], orientation="horizontal", pad=0.06, fraction=0.05, label="mm/day")

    extreme_diff = max_diff - max_gt
    extreme_diff_max = float(np.nanmax(np.abs(extreme_diff)))
    cmap_ed, norm_ed, _ = make_dynamic_cmap(vmax=extreme_diff_max, n_levels=8, kind="diff")
    im = axes[2].imshow(extreme_diff, cmap=cmap_ed, norm=norm_ed, origin="lower")
    axes[2].set_title("Difference (Diffusion \u2212 GT)")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    fig.colorbar(im, ax=axes[2], orientation="horizontal", pad=0.06, fraction=0.05, label="mm/day diff")

    fig.suptitle("Extreme-Day Comparison: Heaviest Rain Day per Pixel, June 1995\n"
                 "(mean-field metrics wash this out \u2014 checks if peak events are captured)")
    plt.savefig(out_path("precip_extreme_day_june1995.png"))
    plt.close()

    # ==========================================
    # METRICS: summary stats
    # ==========================================
    print("Computing summary metrics...")

    def summary_row(name, flat_vals, daily_domain_mean):
        return {
            "dataset": name,
            "domain_mean_mm_day": float(np.mean(flat_vals)),
            "spatial_std_mm_day": float(np.std(flat_vals)),
            "p95_mm_day": float(np.percentile(flat_vals, 95)),
            "p99_mm_day": float(np.percentile(flat_vals, 99)),
            "max_mm_day": float(np.max(flat_vals)),
            "wet_day_frac_pct": wet_day_fraction(daily_domain_mean, thresh=1.0),
        }

    rows = [
        summary_row("GT", flat_gt, spatial_mean_gt),
        summary_row("Diffusion Mean", flat_diff, spatial_mean_diff),
    ]

    print("\n--- Summary metrics (June 1995) ---")
    for r in rows:
        print(r)
    print(f"\nDiffusion Mean vs GT: Bias={stats_diff['bias']:.3f} mm/day, "
          f"RMSE={stats_diff['rmse']:.3f} mm/day, UnbiasedRMSE={stats_diff['unbiased_rmse']:.3f} mm/day, "
          f"Corr={stats_diff['corr']:.3f}, R2={stats_diff['r2']:.3f}")
    print(f"Spread-skill correlation (std vs |error|): {spread_skill_corr:.3f}")
    print(f"Domain-mean Gaussian-approx CRPS: {crps_mean:.3f} mm/day")
    print(f"Taylor diagram: normalized std = {norm_std:.3f}, pattern corr = {pattern_corr:.3f}")

    print("\nFSS (averaged over June days):")
    for thresh in FSS_THRESHOLDS_MMDAY:
        print(f"  >{thresh:g} mm/day: " + ", ".join(
            f"{w}px={v:.3f}" for w, v in zip(FSS_WINDOWS_PX, fss_results[thresh])))

    print("\nContingency scores (pooled over all days/pixels):")
    for thresh, cdf_ in contingency_rows:
        print(f"  >{thresh:g} mm/day: POD={cdf_['pod']:.3f} FAR={cdf_['far']:.3f} CSI={cdf_['csi']:.3f}")

    with open(out_path("metrics_summary_june1995.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_path("metrics_summary_june1995.txt"), "w") as f:
        for r in rows:
            f.write(f"{r}\n")
        f.write(f"\nDiffusion Mean vs GT: Bias={stats_diff['bias']:.3f} mm/day, "
                f"RMSE={stats_diff['rmse']:.3f} mm/day, "
                f"UnbiasedRMSE={stats_diff['unbiased_rmse']:.3f} mm/day, "
                f"Corr={stats_diff['corr']:.3f}, R2={stats_diff['r2']:.3f}, N={stats_diff['n']}\n")
        f.write(f"Spread-skill correlation (ensemble std vs |error|): {spread_skill_corr:.3f}\n")
        f.write(f"Domain-mean Gaussian-approx CRPS: {crps_mean:.3f} mm/day\n")
        f.write(f"Taylor diagram: normalized std = {norm_std:.3f}, pattern corr = {pattern_corr:.3f}\n")

        f.write("\nFSS (averaged over June days):\n")
        for thresh in FSS_THRESHOLDS_MMDAY:
            f.write(f"  >{thresh:g} mm/day: " + ", ".join(
                f"{w}px({w*DX_GT_KM:.0f}km)={v:.3f}"
                for w, v in zip(FSS_WINDOWS_PX, fss_results[thresh])) + "\n")

        f.write("\nContingency scores (pooled over all days/pixels):\n")
        for thresh, cdf_ in contingency_rows:
            f.write(f"  >{thresh:g} mm/day: POD={cdf_['pod']:.3f} FAR={cdf_['far']:.3f} "
                    f"CSI={cdf_['csi']:.3f}\n")

    with open(out_path("fss_summary_june1995.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold_mmday", "window_px", "window_km", "diffusion_fss"])
        for thresh in FSS_THRESHOLDS_MMDAY:
            for i, w in enumerate(FSS_WINDOWS_PX):
                writer.writerow([thresh, w, w * DX_GT_KM, fss_results[thresh][i]])

    with open(out_path("contingency_summary_june1995.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold_mmday", "pod", "far", "csi", "hits", "misses", "false_alarms"])
        for thresh, cdf_ in contingency_rows:
            writer.writerow([thresh, cdf_["pod"], cdf_["far"], cdf_["csi"],
                              cdf_["hits"], cdf_["misses"], cdf_["false_alarms"]])

    # ==========================================
    # FIGURE 16: Summary metric bar chart
    # ==========================================
    print("Generating summary metrics bar chart...")
    metrics_to_plot = ["domain_mean_mm_day", "p95_mm_day", "p99_mm_day", "wet_day_frac_pct"]
    metric_labels = ["Domain Mean", "P95", "P99", "Wet-day Freq (%)"]

    x = np.arange(len(metrics_to_plot))
    width = 0.32

    plt.figure(figsize=(9, 5.5))
    for i, r in enumerate(rows):
        vals = [r[m] for m in metrics_to_plot]
        plt.bar(x + (i - 0.5) * width, vals, width, label=r["dataset"], color=STYLE[r["dataset"]]["color"])

    plt.xticks(x, metric_labels)
    plt.ylabel("Value")
    plt.title("Summary Metric Comparison \u2014 June 1995")
    plt.legend()
    plt.savefig(out_path("precip_metrics_bar_june1995.png"))
    plt.close()

    print(f"\nAll June-1995 GT vs Diffusion-Mean visualizations and metrics saved in '{RESULTS_DIR}/' !")
    ds_gt.close()
    ds_cmp.close()


if __name__ == "__main__":
    main()
