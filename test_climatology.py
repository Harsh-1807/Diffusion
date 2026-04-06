# -*- coding: utf-8 -*-
"""
TEST Climatology Analysis - Regional Focus
Only for Test Dataset (1995-2014 period subset)
All boundaries BLACK except Singapore highlighted
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import geopandas as gpd
from shapely.geometry import box
import warnings
import torch
from torch.utils.data import random_split
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
import math

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
PRECIP_FILE = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/precip_rcm_8km_daily_data_1995-2014.nc"
OUTPUT_DIR = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/CorrDiff/climatology/test_climatology/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "monthly_maps"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "yearly_maps"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)

# Shapefiles
SHAPEFILES = {
    "Singapore": "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_SGP_0.shp",
    "Malaysia": "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_MYS_0.shp",
    "Indonesia": "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_IDN_0.shp",
    "Thailand": "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/ShapeFiles/gadm/gadm41_THA_0.shp",
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def clip_shapefile_to_domain(gdf, lat_min, lat_max, lon_min, lon_max):
    """Clip shapefile to NetCDF domain."""
    bbox = box(lon_min, lat_min, lon_max, lat_max)
    return gdf.clip(bbox)


def load_clipped_shapefiles(lat_min, lat_max, lon_min, lon_max):
    """Load and clip all shapefiles to NetCDF domain."""
    shapefiles = {}
    for country, path in SHAPEFILES.items():
        try:
            if os.path.exists(path):
                gdf = gpd.read_file(path)
                gdf_clipped = clip_shapefile_to_domain(gdf, lat_min, lat_max, lon_min, lon_max)
                if not gdf_clipped.empty:
                    shapefiles[country] = gdf_clipped
                    print(f"✓ {country} (clipped to domain)")
                else:
                    print(f"⚠ {country} (no overlap with domain)")
        except Exception as e:
            print(f"⚠ {country} - Error: {str(e)[:50]}")
    return shapefiles


def make_dynamic_cmap(vmax, n_levels=10, kind="rain"):
    """Dynamic colormap (same style as previous script)."""
    vmax = max(float(vmax), 1e-6)
    raw_step = vmax / max(n_levels - 1, 1)
    magnitude = 10 ** math.floor(math.log10(max(raw_step, 1e-9)))
    step = magnitude * round(raw_step / magnitude)
    if step <= 0:
        step = raw_step

    bounds = np.concatenate(([0], np.arange(step, vmax + step, step)))
    if bounds[-1] < vmax:
        bounds = np.append(bounds, bounds[-1] + step)
    bounds = np.round(bounds, 6)

    base_cmap = plt.get_cmap("jet", len(bounds) - 1)
    colors = base_cmap(np.arange(base_cmap.N))
    colors[0] = [1, 1, 1, 1]           # White for zero
    colors[1] = to_rgba("#6baed6")
    colors[2] = to_rgba("#3182bd")
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm, bounds


def _nice_ticks(vmin, vmax, target=6):
    """Generate clean adaptive ticks."""
    span = vmax - vmin
    raw_step = span / target
    if raw_step == 0:
        return np.array([vmin])
    mag = 10 ** np.floor(np.log10(raw_step))
    step = mag * round(raw_step / mag)
    if step == 0:
        step = raw_step
    ticks = np.arange(vmin, vmax + step, step)
    if ticks[0] > vmin:
        ticks = np.insert(ticks, 0, vmin)
    if ticks[-1] < vmax:
        ticks = np.append(ticks, vmax)
    return np.round(ticks, 4)


def plot_regional_map(data_array, title, output_path, shapefiles, lat_min, lat_max, lon_min, lon_max):
    """Enhanced plot: All boundaries BLACK, Singapore highlighted."""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")

    # Dynamic color scale
    rain_vmax = float(np.nanpercentile(data_array.values, 99.5))
    rain_vmax = max(rain_vmax, 1.0)
    cmap_r, norm_r, _ = make_dynamic_cmap(rain_vmax, n_levels=10)

    # Plot precipitation
    im = ax.pcolormesh(
        data_array.lon.values,
        data_array.lat.values,
        data_array.values,
        cmap=cmap_r,
        norm=norm_r,
        shading='auto',
        zorder=1
    )

    # ====================== BOUNDARIES ======================
    # All countries except Singapore → BLACK
    for country, gdf in shapefiles.items():
        if country == "Singapore":
            continue
        try:
            gdf.boundary.plot(
                ax=ax,
                edgecolor='black',
                linewidth=1.2,
                alpha=0.85,
                zorder=5
            )
        except:
            pass

    # Singapore → Highlighted
    if "Singapore" in shapefiles:
        try:
            shp = shapefiles["Singapore"]
            shp.plot(
                ax=ax,
                facecolor="none",
                edgecolor="#333333",   # Dark charcoal
                linewidth=2.5,
                alpha=1.0,
                zorder=7
            )
        except:
            pass

    # ====================== FORMATTING ======================
    ax.set_xlim([lon_min, lon_max])
    ax.set_ylim([lat_min, lat_max])

    lon_ticks = _nice_ticks(lon_min, lon_max)
    lat_ticks = _nice_ticks(lat_min, lat_max)

    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    ax.tick_params(axis='both', labelsize=11, direction='out', length=4)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3, zorder=4)

    ax.set_xlabel("Longitude (°E)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Latitude (°N)", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Precipitation (mm/day)", fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Legend (non-Singapore countries)
    if len(shapefiles) > 1:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ Saved: {os.path.basename(output_path)}")


# =============================================================================
# MAIN - TEST CLIMATOLOGY
# =============================================================================
def main():
    print("=" * 130)
    print("TEST DATASET CLIMATOLOGY ANALYSIS - REGIONAL FOCUS")
    print("All boundaries BLACK | Singapore highlighted in dark charcoal")
    print("=" * 130)

    # Load full dataset
    print("\n[1/8] Loading precipitation data...")
    ds = xr.open_dataset(PRECIP_FILE)
    pr_full = ds['pr']

    lat_min = float(pr_full.lat.min().values)
    lat_max = float(pr_full.lat.max().values)
    lon_min = float(pr_full.lon.min().values)
    lon_max = float(pr_full.lon.max().values)

    print(f"✓ Full data shape: {pr_full.shape}")
    print(f"✓ Domain: Lat [{lat_min:.2f}–{lat_max:.2f}], Lon [{lon_min:.2f}–{lon_max:.2f}]")

    # Create Test Subset
    print("\n[2/8] Creating Test Dataset subset (15%)...")
    n = len(pr_full.time)
    gen = torch.Generator().manual_seed(42)

    _, _, test_set = random_split(
        range(n),
        [int(0.65 * n), int(0.15 * n), n - int(0.65 * n) - int(0.15 * n)],
        generator=gen
    )
    test_indices = sorted(test_set.indices)

    pr = pr_full.isel(time=test_indices)

    print(f"✓ Test set size: {len(test_indices)} / {n} days ({len(test_indices)/n*100:.1f}%)")
    print(f"✓ Test period: {pd.to_datetime(pr.time.values[0]).date()} to "
          f"{pd.to_datetime(pr.time.values[-1]).date()}")

    # Load shapefiles
    print("\n[3/8] Loading and clipping shapefiles...")
    shapefiles = load_clipped_shapefiles(lat_min, lat_max, lon_min, lon_max)

    # Monthly climatology on Test data
    print("\n[4/8] Computing monthly climatology (Test data only)...")
    monthly_avg = pr.groupby('time.month').mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print("\nMonthly Statistics - TEST DATASET (mm/day):")
    print(f"{'Month':<5} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8} {'P95':>8} {'P99':>8}")
    print("-" * 80)

    monthly_stats = {}
    for month in range(1, 13):
        mean_val = float(monthly_avg.sel(month=month).mean(dim=['lat', 'lon']).values)
        std_val = float(pr.groupby('time.month').std().sel(month=month).mean(dim=['lat', 'lon']).values)
        min_val = float(pr.groupby('time.month').min().sel(month=month).mean(dim=['lat', 'lon']).values)
        max_val = float(pr.groupby('time.month').max().sel(month=month).mean(dim=['lat', 'lon']).values)
        med_val = float(pr.groupby('time.month').median().sel(month=month).mean(dim=['lat', 'lon']).values)
        p95_val = float(pr.groupby('time.month').quantile(0.95).sel(month=month).mean(dim=['lat', 'lon']).values)
        p99_val = float(pr.groupby('time.month').quantile(0.99).sel(month=month).mean(dim=['lat', 'lon']).values)

        monthly_stats[month] = {
            'mean': mean_val, 'std': std_val, 'min': min_val, 'max': max_val,
            'median': med_val, 'p95': p95_val, 'p99': p99_val
        }

        print(f"{months[month-1]:<5} {mean_val:>8.4f} {std_val:>8.4f} {min_val:>8.4f} "
              f"{max_val:>8.4f} {med_val:>8.4f} {p95_val:>8.4f} {p99_val:>8.4f}")

    # Save monthly CSV
    monthly_df = pd.DataFrame({
        'Month': months,
        'Mean (mm/day)': [monthly_stats[m]['mean'] for m in range(1, 13)],
        'Std Dev (mm/day)': [monthly_stats[m]['std'] for m in range(1, 13)],
        'Min (mm/day)': [monthly_stats[m]['min'] for m in range(1, 13)],
        'Max (mm/day)': [monthly_stats[m]['max'] for m in range(1, 13)],
        'Median (mm/day)': [monthly_stats[m]['median'] for m in range(1, 13)],
        'P95 (mm/day)': [monthly_stats[m]['p95'] for m in range(1, 13)],
        'P99 (mm/day)': [monthly_stats[m]['p99'] for m in range(1, 13)],
    })
    monthly_csv = os.path.join(OUTPUT_DIR, "reports", "test_monthly_climatology.csv")
    monthly_df.to_csv(monthly_csv, index=False)
    print(f"\n✓ Saved: {monthly_csv}")

    # Overall Test Statistics
    print("\n[5/8] Overall Test Dataset Statistics...")
    overall_mean = float(pr.mean().values)
    overall_std = float(pr.std().values)

    overall_stats_text = f"""
TEST DATASET STATISTICS (Random Split - ~15% of 1995-2014):
  Number of days : {len(pr.time)} / {n}
  Period         : {pd.to_datetime(pr.time.values[0]).date()} to {pd.to_datetime(pr.time.values[-1]).date()}
  Mean           : {overall_mean:.4f} mm/day
  Std Dev        : {overall_std:.4f} mm/day
  Median         : {float(pr.median().values):.4f} mm/day
  Min            : {float(pr.min().values):.4f} mm/day
  Max            : {float(pr.max().values):.4f} mm/day
  95th Percentile: {float(pr.quantile(0.95).values):.4f} mm/day
  99th Percentile: {float(pr.quantile(0.99).values):.4f} mm/day
"""
    print(overall_stats_text)

    stats_file = os.path.join(OUTPUT_DIR, "reports", "test_overall_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write(overall_stats_text)
    print(f"✓ Saved: {stats_file}")

    # Generate Maps with new style
    print("\n[6/8] Generating enhanced spatial maps (Test Dataset)...")
    spatial_avg = pr.mean(dim='time')
    plot_regional_map(
        spatial_avg,
        f"Test Dataset - Overall Average Precipitation (1995-2014)\nMean: {overall_mean:.4f} mm/day",
        os.path.join(OUTPUT_DIR, "00_test_overall_average.png"),
        shapefiles, lat_min, lat_max, lon_min, lon_max
    )

    print("   → Generating monthly maps (Test data)...")
    for month in range(1, 13):
        spatial_data = monthly_avg.sel(month=month)
        mean_val = monthly_stats[month]['mean']
        plot_regional_map(
            spatial_data,
            f"Test Dataset - {months[month-1]} Average Precipitation\nMean: {mean_val:.4f} mm/day",
            os.path.join(OUTPUT_DIR, "monthly_maps", f"test_{month:02d}_{months[month-1]}.png"),
            shapefiles, lat_min, lat_max, lon_min, lon_max
        )

    # Charts
    print("\n[7/8] Generating charts...")
    monthly_means = [monthly_stats[m]['mean'] for m in range(1, 13)]
    monthly_maxs = [monthly_stats[m]['max'] for m in range(1, 13)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.bar(range(12), monthly_means, color='steelblue', alpha=0.7, edgecolor='navy', linewidth=2, label='Mean')
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(months, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Precipitation (mm/day)', fontsize=12, fontweight='bold')
    ax1.set_title('Monthly Average Precipitation - TEST DATASET', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()

    ax2.bar(range(12), monthly_maxs, color='coral', alpha=0.7, edgecolor='darkred', linewidth=2, label='Max')
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(months, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Maximum Precipitation (mm/day)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax2.set_title('Monthly Maximum Precipitation - TEST DATASET', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "test_monthly_mean_max_chart.png"), dpi=350, bbox_inches='tight')
    plt.close()
    print("✓ Saved: test_monthly_mean_max_chart.png")

    # Summary
    print("\n[8/8] Test Climatology Analysis Complete!")
    print("=" * 130)
    print("SUMMARY - TEST CLIMATOLOGY")
    print("=" * 130)
    print(f"Output Directory : {OUTPUT_DIR}")
    print(f"Test samples     : {len(test_indices)} days (~15%)")
    print(f"Monthly Report   : {monthly_csv}")
    print(f"Overall Report   : {stats_file}")
    print(f"Overall Map      : 00_test_overall_average.png")
    print(f"Monthly Maps     : monthly_maps/ (12 files)")
    print(f"Chart            : test_monthly_mean_max_chart.png")
    print("\nAll boundaries are BLACK except Singapore (highlighted).")
    print("=" * 130 + "\n")


if __name__ == "__main__":
    main()
