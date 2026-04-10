# -*- coding: utf-8 -*-
# Dataset.py  --  FIXED: log-transform BEFORE normalization + TIME TRACKING

import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from datetime import datetime


class ClimateDataset(Dataset):

    VAR_MAP = {
        "huss":   "huss",
        "mslp":   "psl",
        "tas":    "tas",
        "precip": "pr",
    }

    def __init__(self, lr_paths, hr_paths,
                 variables=("huss", "mslp", "tas", "precip"),
                 normalize=True,
                 rank=0):

        self.variables = list(variables)
        self.normalize = normalize
        self._rank     = rank

        def _log(msg):
            if rank == 0:
                print(msg)

        # ---------------- LOAD DATA ----------------
        _log("Loading LR data into RAM...")
        self._lr_arrays = self._load(lr_paths, self.variables)

        _log("Loading HR data into RAM...")
        self._hr_arrays = self._load(hr_paths, self.variables)

        # ============================================================
        # NEW: Extract time information from dataset
        # ============================================================
        _log("Extracting time information from HR dataset...")
        self.years = self._extract_years(hr_paths[0])  # Use first HR file for time coords
        
        assert len(self.years) == self._hr_arrays[self.variables[0]].shape[0], \
            f"Time mismatch: extracted {len(self.years)} years but data has {self._hr_arrays[self.variables[0]].shape[0]} time steps"

        # ---------------- GEO INFO ----------------
        ds_hr = xr.open_dataset(hr_paths[0], decode_times=False)

        if 'lat' in ds_hr:
            self.lats = ds_hr['lat'].values
        elif 'latitude' in ds_hr:
            self.lats = ds_hr['latitude'].values
        else:
            raise KeyError("Latitude variable not found in HR dataset")

        if 'lon' in ds_hr:
            self.lons = ds_hr['lon'].values
        elif 'longitude' in ds_hr:
            self.lons = ds_hr['longitude'].values
        else:
            raise KeyError("Longitude variable not found in HR dataset")

        ds_hr.close()

        # ---------------- LENGTH CHECK ----------------
        T_lr = next(iter(self._lr_arrays.values())).shape[0]
        T_hr = next(iter(self._hr_arrays.values())).shape[0]
        assert T_lr == T_hr, f"Time mismatch: LR={T_lr}  HR={T_hr}"
        self.length = T_hr

        # ============================================================
        # CRITICAL FIX: LOG TRANSFORM BEFORE NORMALIZATION
        # ============================================================
        if "precip" in self.variables:
            idx = self.variables.index("precip")
            _log("Applying log1p to precipitation BEFORE normalization...")

            self._lr_arrays["precip"] = np.log1p(self._lr_arrays["precip"])
            self._hr_arrays["precip"] = np.log1p(self._hr_arrays["precip"])

        # ---------------- NORMALIZATION STATS ----------------
        self.mean = {}
        self.std  = {}

        if normalize:
            _log("Computing normalisation stats from HR...")
            for v in self.variables:
                data   = self._hr_arrays[v].reshape(-1)
                finite = data[np.isfinite(data)]

                self.mean[v] = float(finite.mean())
                self.std[v]  = float(finite.std()) + 1e-6

                _log(f"  {v:8s}: mean={self.mean[v]:.4f}  std={self.std[v]:.4f}")
        else:
            for v in self.variables:
                self.mean[v] = 0.0
                self.std[v]  = 1.0

        # ---------------- NORMALIZE ----------------
        for v in self.variables:
            self._lr_arrays[v] = (
                (self._lr_arrays[v] - self.mean[v]) / self.std[v]
            ).astype(np.float32)

            self._hr_arrays[v] = (
                (self._hr_arrays[v] - self.mean[v]) / self.std[v]
            ).astype(np.float32)

        _log(f"Dataset ready: {self.length} samples  "
             f"LR {next(iter(self._lr_arrays.values())).shape[1:]}  "
             f"HR {next(iter(self._hr_arrays.values())).shape[1:]}")
        
        _log(f"Year range: {min(self.years)} to {max(self.years)}")

    # -----------------------------------------------------------------
    def _extract_years(self, hr_path):
        ds = xr.open_dataset(hr_path)  # IMPORTANT: decode_times=True (default)
    
        if 'time' in ds:
            times = ds['time'].values
        elif 'Time' in ds:
            times = ds['Time'].values
        else:
            raise KeyError("No time coordinate")
    
        # Proper datetime extraction
        years = np.array([t.astype('datetime64[Y]').astype(int) + 1970 for t in times])
    
        ds.close()
        return years

    # -----------------------------------------------------------------
    def _load(self, paths, variables):
        arrays = {}
        for path in paths:
            prefix = path.split("/")[-1].split("_")[0]
            if prefix not in variables:
                continue

            nc_var = self.VAR_MAP[prefix]
            ds     = xr.open_dataset(path, decode_times=False)
            arr    = ds[nc_var].values.astype(np.float32)
            ds.close()

            arrays[prefix] = arr

        return {v: arrays[v] for v in variables}

    # -----------------------------------------------------------------
    def __len__(self):
        return self.length

    # ============================================================
    # NO LOG HERE ANYMORE
    # ============================================================
    def __getitem__(self, idx):
        lr_stack = [self._lr_arrays[v][idx] for v in self.variables]
        hr_stack = [self._hr_arrays[v][idx] for v in self.variables]

        lr = torch.from_numpy(np.stack(lr_stack))
        hr = torch.from_numpy(np.stack(hr_stack))

        return lr.float(), hr.float()


# =============================================================================
# Sanity check
# =============================================================================

if __name__ == "__main__":

    LR_FILES = [
        "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/LR_DATA_32/huss_rcm_8km_daily_data_1995-2014_32km.nc",
        "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/LR_DATA_32/mslp_rcm_8km_daily_data_1995-2014_32km.nc",
        "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/LR_DATA_32/tas_rcm_8km_daily_data_1995-2014_32km.nc",
        "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/LR_DATA_32/precip_rcm_8km_daily_data_1995-2014_32km.nc",
    ]

    HR_FILES = [
        "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/huss_rcm_8km_daily_data_1995-2014.nc",
        "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/mslp_rcm_8km_daily_data_1995-2014.nc",
        "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/tas_rcm_8km_daily_data_1995-2014.nc",
        "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/SRGAN_pipeline/HR_DATA_8/precip_rcm_8km_daily_data_1995-2014.nc",
    ]

    ds = ClimateDataset(LR_FILES, HR_FILES)
    lr, hr = ds[0]

    print("LR:", lr.shape, "HR:", hr.shape)
    for c, name in enumerate(["huss","mslp","tas","precip"]):
        print(f"{name} LR mean {lr[c].mean():.4f} HR mean {hr[c].mean():.4f}")
