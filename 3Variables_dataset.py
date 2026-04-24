import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import warnings
import pandas as pd
import torchvision.transforms.functional as TF
import random

warnings.filterwarnings("ignore")

class UpscaleDataset(torch.utils.data.Dataset):
    PRECIPITATION_NAMES = [
        'rf', 'RF', 'rainfall', 'RAINFALL', 'precipitation', 'PRECIPITATION',
        'pr', 'PR', 'tp', 'TP', 'total_precipitation',
        'prec', 'PREC', 'rain', 'RAIN', 'precip', 'PRECIP'
    ]

    def __init__(
        self,
        nc_file,
        oro_file,
        d2m_file=None,           # NEW: path to era5_aligned_to_rf.nc
        downscale_factor=4,
        normalize=True,
        device="cuda",
        auto_detect_var=True,
        variable_name=None,
        split='train',
        train_frac=0.65,
        coarse_qdm=None,
    ):
        self.split = split
        self.downscale = downscale_factor
        self.has_d2m = d2m_file is not None

        # ---------- LOAD PRECIP ----------
        try:
            ds = xr.open_dataset(nc_file, engine="netcdf4")
        except Exception:
            ds = xr.open_dataset(nc_file, engine="h5netcdf")

        ds_oro = xr.open_dataset(oro_file, engine="netcdf4")
        topo = ds_oro['topology'].values.astype(np.float32)
        topo = np.nan_to_num(topo, nan=0.0)

        # ---------- LOAD D2M ----------
        if self.has_d2m:
            ds_d2m = xr.open_dataset(d2m_file, engine="netcdf4")
            # try common var names
            d2m_var = None
            for v in ['d2m', 'D2M', 'dewpoint_temperature_2m', 'dew_point_temperature']:
                if v in ds_d2m.data_vars:
                    d2m_var = v; break
            if d2m_var is None:
                d2m_var = list(ds_d2m.data_vars)[0]
            d2m_raw = ds_d2m[d2m_var].values.astype(np.float32)  # [T, H, W] or [T, 1, H, W]
            if d2m_raw.ndim == 4:
                d2m_raw = d2m_raw[:, 0]   # drop channel dim if present
            d2m_raw = np.nan_to_num(d2m_raw, nan=0.0)
            # convert K -> C if values look like Kelvin
            if d2m_raw.mean() > 100:
                d2m_raw = d2m_raw - 273.15

        # ---------- VARIABLE SELECTION ----------
        if variable_name is not None:
            data = ds[variable_name].values
        elif auto_detect_var:
            found = None
            for v in ds.data_vars:
                if any(k.lower() in v.lower() for k in self.PRECIPITATION_NAMES):
                    found = v; break
            if found is None:
                raise ValueError("No precipitation variable detected")
            data = ds[found].values
        else:
            data = ds[list(ds.data_vars)[0]].values

        data = data.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        data = np.clip(data, a_min=0.0, a_max=None)

        dt = pd.to_datetime(ds["TIME"].values)
        self.doy  = torch.tensor(dt.dayofyear.values / 366.0, dtype=torch.float32)
        self.hour = torch.tensor(dt.hour.values / 24.0,       dtype=torch.float32)

        # ---------- TOPO NORM ----------
        self.log_transformed = normalize
        topo_mean = topo.mean(); topo_std = topo.std() + 1e-8
        if normalize:
            topo = (topo - topo_mean) / topo_std
        self.topo_mean = topo_mean; self.topo_std = topo_std

        # ---------- D2M NORM (z-score — best for left-skewed bounded var) ----------
        # D2M ~ [-40, 25] °C, left-skewed. Z-score centers & scales without
        # distorting tails; diffusion model sees near-Gaussian input.
        if self.has_d2m and normalize:
            self.d2m_mean = float(np.nanmean(d2m_raw))
            self.d2m_std  = float(np.nanstd(d2m_raw) + 1e-8)
            d2m_raw = (d2m_raw - self.d2m_mean) / self.d2m_std
        elif self.has_d2m:
            self.d2m_mean = 0.0; self.d2m_std = 1.0

        # ---------- CROP TO MULTIPLE OF 16 ----------
        H, W = data.shape[1], data.shape[2]
        H = (H // 16) * 16; W = (W // 16) * 16
        data = data[:, :H, :W]
        topo = topo[:H, :W]
        if self.has_d2m:
            d2m_raw = d2m_raw[:, :H, :W]

        self.H, self.W = H, W
        T = data.shape[0]
        self.doy  = self.doy[:T]
        self.hour = self.hour[:T]
        self.topo_tensor = torch.from_numpy(topo).unsqueeze(0).contiguous()  # [1,H,W]

        if self.has_d2m:
            self.d2m_tensor = torch.from_numpy(d2m_raw).unsqueeze(1).contiguous()  # [T,1,H,W]

        # ---------- DEVICE ----------
        if device == "cuda" and torch.cuda.is_available():
            compute_device = torch.device("cuda")
        else:
            compute_device = torch.device("cpu")

        print(f"Dataset Split: {self.split}")
        print(f"Fine resolution: {self.H}x{self.W}")
        print(f"Coarse resolution: {self.H//downscale_factor}x{self.W//downscale_factor}")
        print(f"D2M channel: {'YES' if self.has_d2m else 'NO'}")

        # ---------- PRECOMPUTE TENSORS ----------
        fine_all, coarse_all = [], []
        batch_size = 32

        for i in range(0, T, batch_size):
            chunk = torch.from_numpy(data[i:i+batch_size]).unsqueeze(1)
            if compute_device.type == "cuda":
                chunk = chunk.to(compute_device, non_blocking=True)
            coarse = F.avg_pool2d(chunk, kernel_size=4, stride=4)
            if self.log_transformed:
                chunk  = torch.log1p(chunk)
                coarse = torch.log1p(coarse)
            fine_all.append(chunk.cpu())
            coarse_all.append(coarse.cpu())

        self.fine   = torch.cat(fine_all).contiguous()    # [T,1,H,W]
        self.coarse = torch.cat(coarse_all).contiguous()  # [T,1,H/4,W/4]

        if coarse_qdm is not None:
            self.coarse_qdm = coarse_qdm
        else:
            self.coarse_qdm = self.coarse.clone()

    def __len__(self):
        return self.fine.shape[0]

    def __getitem__(self, idx):
        fine       = self.fine[idx]           # [1,H,W]
        coarse     = self.coarse[idx]         # [1,H/4,W/4]
        coarse_qdm = self.coarse_qdm[idx]
        doy        = self.doy[idx]
        hour       = self.hour[idx]
        topo       = self.topo_tensor.clone() # [1,H,W]

        # d2m at fine resolution [1,H,W]
        d2m = self.d2m_tensor[idx].clone() if self.has_d2m else None

        if self.split == 'train':
            if random.random() > 0.5:
                fine = TF.hflip(fine); coarse = TF.hflip(coarse)
                coarse_qdm = TF.hflip(coarse_qdm); topo = TF.hflip(topo)
                if d2m is not None: d2m = TF.hflip(d2m)

            if random.random() > 0.5:
                fine = TF.vflip(fine); coarse = TF.vflip(coarse)
                coarse_qdm = TF.vflip(coarse_qdm); topo = TF.vflip(topo)
                if d2m is not None: d2m = TF.vflip(d2m)

            angle = random.choice([0, 180])
            if angle != 0:
                fine = TF.rotate(fine, angle); coarse = TF.rotate(coarse, angle)
                coarse_qdm = TF.rotate(coarse_qdm, angle); topo = TF.rotate(topo, angle)
                if d2m is not None: d2m = TF.rotate(d2m, angle)

        out = {
            "fine": fine, "coarse": coarse, "coarse_qdm": coarse_qdm,
            "topo": topo, "doy": doy, "hour": hour,
            "idx": torch.tensor(idx, dtype=torch.long),
        }
        if d2m is not None:
            out["d2m"] = d2m
        return out

    def denormalize(self, data):
        if self.log_transformed:
            return torch.expm1(data)
        return data

    def denormalize_d2m(self, data):
        """Reverse z-score on d2m."""
        return data * self.d2m_std + self.d2m_mean
