import os
import numpy as np
import pandas as pd
import torch
import xarray as xr
from tqdm import tqdm
from scipy.spatial import cKDTree

from Dataset import ClimateDataset
from Network import DiffusionUNet
from Regressor import ClimateRegressor


# ================= PATHS =================
LR_FILES = [
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/gt_extracted/huss_rcm_8km_daily_data_2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/gt_extracted/mslp_rcm_8km_daily_data_2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/gt_extracted/tas_rcm_8km_daily_data_2014.nc",
    "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/gt_extracted/precip_rcm_8km_daily_data_2014.nc",
]

BEST_CKPT = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/CorrDiff/2km/corrdiff_singapore_v7b_latest.pth"
REG_CKPT  = "regressor.pth"

STATION_CSV = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/data_exploration/CODE/rainfall_station_final.csv"
TS_BASE_DIR = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/data_exploration/OUTPUT/Station_timeseries/"

OUTPUT_FILE = "model_vs_station.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRECIP_CH = 3
N_INFER = 50


# ================= SAFE LOAD =================
def _torch_load(path, device):
    import numpy as np
    import numpy._core.multiarray

    SAFE = [
        np.ndarray, np.dtype,
        numpy._core.multiarray._reconstruct,
        numpy._core.multiarray.scalar,
    ]

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except:
        pass

    try:
        with torch.serialization.safe_globals(SAFE):
            return torch.load(path, map_location=device, weights_only=True)
    except:
        pass

    return torch.load(path, map_location=device, weights_only=False)


# ================= MODEL =================
class SingaporeCorrDiffNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = DiffusionUNet(in_channels=12, out_channels=4)

    def forward(self, x, t):
        return self.net(x, t)


class SingaporeCorrDiffModel(torch.nn.Module):
    def __init__(self, diffusion_net, regressor):
        super().__init__()
        self.diffusion_net = diffusion_net
        self.regressor = regressor

    def get_mu(self, lr):
        with torch.no_grad():
            return self.regressor(lr)

    @torch.no_grad()
    def sample(self, lr, n_steps=50):
        return self.get_mu(lr)


def load_model():
    ckpt = _torch_load(BEST_CKPT, DEVICE)

    reg = ClimateRegressor().to(DEVICE)
    reg.load_state_dict(_torch_load(REG_CKPT, DEVICE))
    reg.eval()

    dnet = SingaporeCorrDiffNet().to(DEVICE)
    model = SingaporeCorrDiffModel(dnet, reg).to(DEVICE)

    raw = ckpt.get("model_state_dict", ckpt)
    clean = {k.replace("module.", ""): v for k, v in raw.items()}
    model.diffusion_net.load_state_dict(clean, strict=False)

    model.eval()
    return model


# ================= NORMALIZATION =================
def to_mm(arr_norm, mean, std):
    x_log = arr_norm * std + mean
    return np.expm1(x_log).clip(0)


# ================= LOAD DATA =================
dataset = ClimateDataset(LR_FILES, LR_FILES, rank=0)
PRECIP_MEAN = dataset.mean["precip"]
PRECIP_STD  = dataset.std["precip"]

ds = xr.open_dataset(LR_FILES[3])
time_array = pd.to_datetime(ds["time"].values)
ds.close()


# ================= STATIONS =================
stations = pd.read_csv(STATION_CSV)

def build_folder_name(row):
    return f"{row['Station ID']}_{row['Station Name'].replace(' ', '_')}"

station_data = {}

for _, row in stations.iterrows():
    folder = os.path.join(TS_BASE_DIR, build_folder_name(row))
    file = os.path.join(folder, f"{build_folder_name(row)}_timeseries.csv")

    if not os.path.exists(file):
        continue

    df = pd.read_csv(file)
    df["datetime"] = pd.to_datetime(df["date"])
    station_data[row["Station ID"]] = df.set_index("datetime")["rainfall"]


# ================= GRID =================
lats = dataset.lats
lons = dataset.lons

def make_latlon(shape):
    H, W = shape
    lat_hr = np.linspace(lats.min(), lats.max(), H)
    lon_hr = np.linspace(lons.min(), lons.max(), W)
    return lat_hr, lon_hr


# ================= RUN =================
model = load_model()

results = []

print("Running...")

for idx in tqdm(range(len(dataset))):

    date = time_array[idx]

    lr_s, _ = dataset[idx]
    lr_t = lr_s.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model.sample(lr_t, n_steps=N_INFER)

    pred_mm = to_mm(pred[0, PRECIP_CH].cpu().numpy(),
                    PRECIP_MEAN, PRECIP_STD)

    lat_hr, lon_hr = make_latlon(pred_mm.shape)

    # ✅ FAST + CORRECT KDTree
    lat_grid, lon_grid = np.meshgrid(lat_hr, lon_hr, indexing="ij")
    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    tree = cKDTree(pts)

    for _, row in stations.iterrows():

        sid = row["Station ID"]

        lat_s = row["Latitude"]
        lon_s = row["Longitude"]

        # ✅ FIX: skip invalid coords
        if not np.isfinite(lat_s) or not np.isfinite(lon_s):
            continue

        if sid not in station_data:
            continue

        if date not in station_data[sid].index:
            continue

        obs = station_data[sid].loc[date]

        dist, idx_nn = tree.query([lat_s, lon_s])

        i = idx_nn // len(lon_hr)
        j = idx_nn % len(lon_hr)

        pred_val = pred_mm[i, j]

        results.append({
            "date": date,
            "station": sid,
            "observed": obs,
            "predicted": pred_val
        })


# ================= SAVE =================
pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
