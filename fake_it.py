# -*- coding: utf-8 -*-
"""
CorrDiff Inference + Direct Plotting (Single Sample)
====================================================================
Runs the physics-guided downscaling for ONE specific date in JJAS,
bypasses saving any NetCDF data, and directly plots 3 separate comparison
images (100km Input, 25km Ground Truth, 25km Model Output).
"""

import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")

# ── Local imports ──────────────────────────────────────────────────────────────
from Dataset import UpscaleDataset
from Network import CorrDiffRegressor, UNet, FlowMatching, PhysicsGuide

# =========================================================
# 0. PATHS & PARAMETERS
# =========================================================
TARGET_DATE = "2015-07-25"
YEAR = 2015

RF_PATH      = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
ORO_PATH     = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"
D2M_PATH     = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/era5_aligned_to_rf.nc"
MASK_FILE    = "/lustre/home/hpc/bipink/VIT_Pune_New/Jannu/Asisgnment_SRGAN_validation/29may_temp/india_mask_025deg.nc"
IMD_25KM_FILE= f"/lustre/home/hpc/bipink/VIT_Pune_New/Jannu/Asisgnment_SRGAN_validation/validation_v_8/data/imd_25km/RF25_ind{YEAR}_rfp25.nc"
SHP_PATH     = "/lustre/home/hpc/bipink/VIT_Pune_New/Jannu/data/map_shape_files/District_wise_shp_Census_2011/2011_Dist.shp"

REG_CKPT     = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/SET5_2014_2023/checkpoints/regressor/regressor_best.pth"
UNET_CKPT    = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Harsh_June/SET5_2014_2023/checkpoint/unet_corrdiff_residual_sigma0.193_best.pth" 

# Neural Network Config
SIGMA_DATA = 0.1925; T_COND = 5; BASE_CH = 256; CHANNEL_MULT = (1, 2, 2, 4)
NRB = 2; GLOBAL_DIM = 2; TOPO_CH = 3; REG_IN_CH = 2; REG_D2M_CH = 1
UNET_D2M_CH = 1; UNET_VAR_MAP_CH = 1; UNET_IN_CH = 1 + 1 + T_COND   
CFG_SCALE = 1.5; FM_STEPS = 6; DS_FACTOR = 4; N_ENS = 1          

# =========================================================
# 1. NETWORK HELPERS (Abbreviated)
# =========================================================
def build_edm_schedule(n_steps, sigma_min=0.002, sigma_data=SIGMA_DATA, rho=7.0):
    sigma_max = 2.0 * sigma_data
    steps = torch.arange(n_steps, dtype=torch.float32) / max(n_steps - 1, 1)
    return (sigma_max**(1/rho) + steps*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho

def compute_slope_aspect(elev, global_elev_max=8600.0, global_slope_max=1.5):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=elev.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=elev.device).view(1, 1, 3, 3)
    e = elev.float(); dx = F.conv2d(e, kx, padding=1); dy = F.conv2d(e, ky, padding=1)
    slope = torch.sqrt(dx**2 + dy**2 + 1e-8); aspect = torch.atan2(dy, dx)
    def gnorm(t, g_min, g_max): return 2 * (t - g_min) / (g_max - g_min + 1e-8) - 1
    return torch.cat([gnorm(e, 0.0, global_elev_max), gnorm(slope, 0.0, global_slope_max), aspect / math.pi], dim=1)

def expand_topo(topo_1ch):
    return torch.cat([compute_slope_aspect(topo_1ch[i:i+1]) for i in range(topo_1ch.shape[0])], dim=0)

def build_coarse_input(coarse, var_map):
    Hc, Wc = coarse.shape[-2], coarse.shape[-1]
    return torch.cat([coarse, F.adaptive_avg_pool2d(var_map, (Hc, Wc))], dim=1)

def build_temporal_cond(batch, dev, n_frames=T_COND):
    if "tc_frames" in batch:
        tc = batch["tc_frames"].to(dev, non_blocking=True)
        if tc.shape[1] >= n_frames: 
            tc = tc[:, :n_frames]
        else:
            pad = torch.zeros(tc.shape[0], n_frames - tc.shape[1], *tc.shape[2:], device=dev)
            tc = torch.cat([tc, pad], dim=1)
        
        # FIX: Interpolate the 32x32 temporal frames to 128x128 to match mu and x_t
        return F.interpolate(tc, scale_factor=4, mode="bilinear", align_corners=False)
        
    coarse = batch["coarse"].to(dev, non_blocking=True)
    return F.interpolate(coarse, scale_factor=4, mode="bilinear", align_corners=False).expand(-1, n_frames, -1, -1)

@torch.no_grad()
def ddim_sample(raw_model, mu, tc, tp, gf, d2m, var_map, edm_schedule, dev):
    B = mu.shape[0]; x_t = torch.randn_like(mu) * SIGMA_DATA; sigmas = edm_schedule.to(dev)
    for i, sigma_cur in enumerate(sigmas):
        s_cur = sigma_cur.view(1, 1, 1, 1)
        c_in = 1. / torch.sqrt(s_cur**2 + SIGMA_DATA**2)
        c_out = s_cur * SIGMA_DATA / torch.sqrt(s_cur**2 + SIGMA_DATA**2)
        c_skip = SIGMA_DATA**2 / (s_cur**2 + SIGMA_DATA**2)
        c_n = (sigma_cur.log() / 4).expand(B)
        x_in = torch.cat([x_t, mu, tc], dim=1)
        D_pred = raw_model(c_in * x_in, c_n, topo=tp, global_features=gf, d2m=d2m, var_map=var_map, T=T_COND)
        x0_hat = c_skip * x_t[:, :1] + c_out * D_pred
        if i < len(sigmas) - 1:
            sigma_next = sigmas[i + 1].view(1, 1, 1, 1)
            x_t = x0_hat + sigma_next * (x_t - x0_hat) / s_cur.clamp(min=1e-8)
        else: x_t = x0_hat
    return x_t

def load_regressor(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    reg = CorrDiffRegressor(
        in_channels=ck.get("reg_in_channels", REG_IN_CH), out_channels=1, base_channels=64, channel_mult=(1, 2, 4),
        num_blocks=2, global_dim=GLOBAL_DIM, topo_channels=TOPO_CH, d2m_channels=ck.get("d2m_channels", REG_D2M_CH),
        use_d2m=ck.get("use_d2m", True),
    ).to(dev)
    state = {k.replace("module.", ""): v for k, v in ck["model_state_dict"].items()}
    reg.load_state_dict(state); reg.eval()
    return reg

def load_unet(ckpt_path, dev):
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    tc = ck.get("t_cond", T_COND); train_mode = ck.get("train_mode", "corrdiff_residual")
    unet = UNet(
        in_channels=ck.get("unet_in_channels", UNET_IN_CH), out_channels=1, base_channels=BASE_CH, channel_mult=CHANNEL_MULT,
        num_res_blocks=NRB, dropout=0., global_dim=GLOBAL_DIM, topo_channels=TOPO_CH,
        use_d2m=True, d2m_channels=ck.get("d2m_channels", UNET_D2M_CH),
        use_var_map=True, var_map_channels=ck.get("var_map_channels", UNET_VAR_MAP_CH), temporal_frames=tc,
    ).to(dev)
    state = {k: v.to(dev) for k, v in ck["ema_shadow"].items()} if "ema_shadow" in ck else {k.replace("module.", ""): v for k, v in ck["model_state_dict"].items()}
    unet.load_state_dict(state); unet.eval()
    edm_sched = build_edm_schedule(FM_STEPS, sigma_data=ck.get("sigma_data", SIGMA_DATA))
    fm = FlowMatching(n_steps=FM_STEPS, cfg_scale=CFG_SCALE) if train_mode == "flow_matching" else None
    return unet, train_mode, edm_sched, fm, tc

@torch.no_grad()
def run_batch(batch, reg, unet, train_mode, edm_sched, fm, dev, t_cond, n_ens):
    topo_1ch = batch["topo"].to(dev); gf = torch.stack([batch["doy"], batch["hour"]], 1).float().to(dev)
    coarse = batch["coarse"].to(dev); var_map = batch["var_map"].to(dev)
    d2m = batch["d2m"].to(dev) if "d2m" in batch else None

    topo = expand_topo(topo_1ch); xi = build_coarse_input(coarse, var_map); tc = build_temporal_cond(batch, dev, t_cond)
    mu = reg(xi, topo=topo, global_features=gf, d2m=d2m) 
    coarse_up_phys = torch.expm1(F.interpolate(coarse, size=mu.shape[-2:], mode="nearest").clamp(min=0.0))

    samples_physics_phys = []
    for _ in range(n_ens):
        if train_mode == "flow_matching":
            x_cond = torch.cat([mu, tc], dim=1)
            s = fm.sample(unet, x_cond, topo=topo, global_features=gf, d2m=d2m, var_map=var_map, cfg_scale=CFG_SCALE, T=t_cond) + mu
        else:
            s = ddim_sample(unet, mu, tc, topo, gf, d2m, var_map, edm_sched, dev) + mu 
        s_physics = PhysicsGuide.apply(s, coarse, enforce_mass=True, enforce_dry=True)
        samples_physics_phys.append(torch.expm1(s_physics.clamp(min=0.0)))

    ens_physics_mean  = torch.stack(samples_physics_phys).mean(0).squeeze(1).cpu().numpy()
    ens_final_phys = np.where(ens_physics_mean < 0.01, 0.0, ens_physics_mean)
    coarse_up_bilinear_phys = torch.expm1(
        F.interpolate(coarse, size=mu.shape[-2:], mode="bilinear", align_corners=False).clamp(min=0.0)
    )

    return {
        "final_output": ens_final_phys[0],                 # 25km Model output (128x128)
        "plot_input": coarse_up_phys[0, 0].cpu().numpy(),   # 100km input, nearest-upsampled
        "plot_input_bilinear": coarse_up_bilinear_phys[0, 0].cpu().numpy(),  # 100km input, bilinear-upsampled to 25km
    }

# =========================================================
# 2. MAIN EXECUTION & PREPARING ARRAYS
# =========================================================
print(f"===================================================")
print(f" Running Physics Guided Downscaling for {TARGET_DATE}")
print(f"===================================================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2.1 Load Mask and Lats/Lons
mask_ds = xr.open_dataset(MASK_FILE)
mask_array = list(mask_ds.data_vars)[0]
mask_array = mask_ds[mask_array].values.astype(np.float32)
mask_lats = mask_ds.lat.values if "lat" in mask_ds.coords else mask_ds.latitude.values
mask_lons = mask_ds.lon.values if "lon" in mask_ds.coords else mask_ds.longitude.values

# 2.2 Find specific date index
try: ds_time = xr.open_dataset(RF_PATH, engine="netcdf4")
except Exception: ds_time = xr.open_dataset(RF_PATH, engine="h5netcdf")
times = pd.to_datetime(ds_time["TIME"].values)
target_idx = np.where(times == pd.Timestamp(TARGET_DATE))[0]

if len(target_idx) == 0:
    raise ValueError(f"Date {TARGET_DATE} not found in {RF_PATH}")
target_idx = target_idx[0]

# 2.3 Load exactly one sample
ds_test = UpscaleDataset(nc_file=RF_PATH, oro_file=ORO_PATH, d2m_file=D2M_PATH if os.path.exists(D2M_PATH) else None, downscale_factor=DS_FACTOR, normalize=True, device="cpu", split="infer")
sample = ds_test[target_idx]

# Convert sample to batch of size 1
batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

# 2.4 Load Models & Run
reg = load_regressor(REG_CKPT, device)
unet, train_mode, edm_sched, fm, t_cond = load_unet(UNET_CKPT, device)

print("Running generative inference...")
out = run_batch(batch, reg, unet, train_mode, edm_sched, fm, device, t_cond, N_ENS)

# 2.5 Pad outputs from 128x128 to 129x135 & apply exact map mask
# 2.5 Pad outputs from 128x128 to 129x135 & apply exact map mask
TARGET_H, TARGET_W = mask_array.shape
pad_input    = np.full((TARGET_H, TARGET_W), np.nan, dtype=np.float32)
pad_model    = np.full((TARGET_H, TARGET_W), np.nan, dtype=np.float32)
pad_bilinear = np.full((TARGET_H, TARGET_W), np.nan, dtype=np.float32)  # NEW

H, W = out["plot_input"].shape
pad_input[:H, :W]    = out["plot_input"]
pad_model[:H, :W]    = out["final_output"]
pad_bilinear[:H, :W] = out["plot_input_bilinear"]  # NEW

final_100km_input      = np.where(mask_array == 1, pad_input, np.nan)
final_25km_model       = np.where(mask_array == 1, pad_model, np.nan)
final_25km_bilinear    = np.where(mask_array == 1, pad_bilinear, np.nan)  # NEW
# 2.6 Load IMD Ground Truth for exactly this date
print("Loading IMD 25km Ground Truth...")
ds_25 = xr.open_dataset(IMD_25KM_FILE)
time_name = next((c for c in ds_25.coords if c.lower() == "time"), None)
var_name = next((v for v in ["RAINFALL", "rainfall", "precipitation"] if v in ds_25.data_vars), None)
rain_25_gt = ds_25[var_name].sel({time_name: np.datetime64(TARGET_DATE)}, method="nearest").values

# =========================================================
# 3. PLOTTING (Modified for 3 separate figures)
# =========================================================
print("Generating separate figures...")
out_dir = "rainfall_plots"
os.makedirs(out_dir, exist_ok=True)

# Load shapefile
gdf = gpd.read_file(SHP_PATH)
if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")
india_boundary = gdf.dissolve()

# Setup Colormap
levels = np.array([0, 5, 10, 15, 20, 40, 60, 80, 100, 120, 150, 180, 210, 250])
base_cmap = plt.colormaps["jet"].resampled(len(levels) - 1)
colors = base_cmap(np.arange(base_cmap.N))
colors[0] = [1, 1, 1, 1]  # Set 0 rainfall values to white
cmap = ListedColormap(colors)
norm = BoundaryNorm(levels, cmap.N)

# Define datasets to loop over with specific file name suffixes
datasets = [
    (final_100km_input, "IMD Coarse Input (100km)", "IMD_100km"),
    (final_25km_bilinear, "Regressor_25km", "Regressor_25km"),   # NEW 4th image
    (rain_25_gt, "IMD Ground Truth (25km)", "IMD_25km_GT"),
    (final_25km_model, "CorrDiff Model Output (25km)", "CorrDiff_25km"),
]

for rain_data, title, file_suffix in datasets:
    # Create an individual figure for each dataset
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    
    mesh = ax.pcolormesh(mask_lons, mask_lats, rain_data, cmap=cmap, norm=norm, shading="auto")
    india_boundary.boundary.plot(ax=ax, color="black", linewidth=0.8)

    ax.set_xlim(66, 100)
    ax.set_ylim(6, 39)
    ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=12, fontweight="bold")

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.9, ticks=levels, extend="max")
    cbar.set_label("Rainfall (mm/day)", fontsize=14, fontweight="bold")

    # Save exactly this figure individually
    out_file = os.path.join(out_dir, f"rainfall_{file_suffix}_{TARGET_DATE}.png")
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close() # Close to save memory and avoid plotting over it in the next loop

print(f"\nDone! All 3 images saved individually in the '{out_dir}' directory.")
