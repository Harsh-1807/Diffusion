Dataset Split: train
Fine resolution: 128x128
Coarse resolution: 32x32
Dataset Split: train
Fine resolution: 128x128
Coarse resolution: 32x32
Dataset Split: train
Fine resolution: 128x128
Coarse resolution: 32x32
Dataset Split: train
Fine resolution: 128x128
Coarse resolution: 32x32
Split  Train:12527  Val:1789

Fitting QDM (1500 quantiles/pixel, linear mm/day)...
  4000 timesteps | 32x32 grid
  Coarse: mean=0.641  p95=2.72  max=225.48 mm/day
  OBS-LR: mean=0.970  p95=5.13  max=278.04 mm/day
  Q50 shift (obs-coarse): 0.0000 mm/day  (38s)
  Saved to checkpoints/diffusion/qdm_tables.npz

QDM sanity (mm/day):
  coarse [0.00, 113.21]  obs-LR [0.00, 129.14]  mapped [0.00, 98.40]
  Raw MAE=0.4231  QDM MAE=0.2570  Improvement=+0.1661  (better)

Regressor frozen.  sigma_data=0.2098

UNet: 110.86M params
   Ep |    Ledm |    Lint | wPCC_tr | wPCC_val | s
----------------------------------------------------






Loading LR data into RAM...
Loading HR data into RAM...
Applying log1p to precipitation BEFORE normalization...
Computing normalisation stats from HR...
  huss    : mean=17.4145  std=1.4961
  mslp    : mean=1009.5259  std=1.5250
  tas     : mean=27.5557  std=1.6956
  precip  : mean=0.1499  std=0.3729
Dataset ready: 7305 samples  LR (64, 69)  HR (256, 276)
Dataset: 7305 Train 5113 Val 1096 Test 1096
WET_THRESH_NORM = -0.4018
Singapore weight map: 256x276 peak=12.99x
  Computing PSD intersection scale s ...
  s_freq = 0.0781
 Collecting training data for QDM fitting...
  Fitting GLOBAL + LOCAL QDM...
  QDM fitting complete.
 Embedding ready: s_freq=0.0781
Estimating sigma_data ...
 sigma_data=1.4146 sigma_min=0.0071 sigma_max=2.2634

======================================================================
CorrDiff Singapore v7b — SURGICAL MINIMAL FIX
  Fix A: SSIM C1/C2 now fixed constants (was data-range → saturated)
  Fix B: intensity += pinball@0.95,0.99 (nudge, not replace)
  Fix C: adaptive weight update unified bounds (no duplicate floor)
  Fix D: 'tail' key always in metrics dict
  LR: 1e-6 (conservative for epoch ~1580 fine-tuning)
  Scheduler: CosineAnnealing T_max=2000 (no warm-restart spike)
  Peak SG weight = 13.0x
  Resume epoch = 0
======================================================================

Ep     1 | lr 1.00e-06 | loss 9.7005 [edm 0.556(1.68) pcc 0.268(7.00) int 1.193(3.00) ssim 0.459(0.32) tail 0.114] | val PCC 0.6753 SG-PCC 0.5131 I-MAE 2.3327 | composite 0.4159 | 108.9s
 >> NEW BEST composite=0.4159 (PCC=0.6753 SG-PCC=0.5131 I-MAE=2.3327)
Ep     2 | lr 1.00e-06 | loss 10.7875 [edm 0.545(2.00) pcc 0.256(7.00) int 1.233(3.00) ssim 0.458(0.38) tail 0.114] | val PCC 0.6814 SG-PCC 0.5165 I-MAE 2.2738 | composite 0.4089 | 107.7s
 >> NEW BEST composite=0.4089 (PCC=0.6814 SG-PCC=0.5165 I-MAE=2.2738)
Ep     3 | lr 1.00e-06 | loss 10.7834 [edm 0.546(2.00) pcc 0.257(7.00) int 1.193(3.00) ssim 0.458(0.42) tail 0.126] | val PCC 0.6900 SG-PCC 0.5242 I-MAE 2.2735 | composite 0.4032 | 108.6s
 >> NEW BEST composite=0.4032 (PCC=0.6900 SG-PCC=0.5242 I-MAE=2.2735)
Ep     4 | lr 1.00e-06 | loss 10.0900 [edm 0.559(2.00) pcc 0.246(7.00) int 1.197(3.00) ssim 0.457(0.44) tail 0.111] | val PCC 0.7003 SG-PCC 0.5315 I-MAE 2.2658 | composite 0.3964 | 107.9s
 >> NEW BEST composite=0.3964 (PCC=0.7003 SG-PCC=0.5315 I-MAE=2.2658)
Ep     5 | lr 1.00e-06 | loss 10.1777 [edm 0.568(2.00) pcc 0.224(7.00) int 1.202(3.00) ssim 0.455(0.46) tail 0.089] | val PCC 0.7113 SG-PCC 0.5392 I-MAE 2.2157 | composite 0.3867 | 107.6s
 >> NEW BEST composite=0.3867 (PCC=0.7113 SG-PCC=0.5392 I-MAE=2.2157)
Ep     6 | lr 1.00e-06 | loss 9.6903 [edm 0.577(2.00) pcc 0.224(7.00) int 1.203(3.00) ssim 0.455(0.47) tail 0.094] | val PCC 0.7214 SG-PCC 0.5484 I-MAE 2.2111 | composite 0.3796 | 108.0s
 >> NEW BEST composite=0.3796 (PCC=0.7214 SG-PCC=0.5484 I-MAE=2.2111)
Ep     7 | lr 1.00e-06 | loss 9.4673 [edm 0.571(2.00) pcc 0.221(7.00) int 1.153(3.00) ssim 0.456(0.49) tail 0.116] | val PCC 0.7302 SG-PCC 0.5560 I-MAE 2.1129 | composite 0.3679 | 108.2s
 >> NEW BEST composite=0.3679 (PCC=0.7302 SG-PCC=0.5560 I-MAE=2.1129)
Ep     8 | lr 1.00e-06 | loss 9.9611 [edm 0.574(2.00) pcc 0.221(7.00) int 1.171(3.00) ssim 0.455(0.49) tail 0.074] | val PCC 0.7369 SG-PCC 0.5636 I-MAE 2.1035 | composite 0.3624 | 108.1s
 >> NEW BEST composite=0.3624 (PCC=0.7369 SG-PCC=0.5636 I-MAE=2.1035)
Ep     9 | lr 1.00e-06 | loss 9.0017 [edm 0.568(2.00) pcc 0.215(7.00) int 1.112(3.00) ssim 0.455(0.50) tail 0.071] | val PCC 0.7427 SG-PCC 0.5718 I-MAE 2.0853 | composite 0.3565 | 107.7s
 >> NEW BEST composite=0.3565 (PCC=0.7427 SG-PCC=0.5718 I-MAE=2.0853)
Ep    10 | lr 1.00e-06 | loss 9.5018 [edm 0.556(2.00) pcc 0.218(7.00) int 1.094(3.00) ssim 0.456(0.50) tail 0.083] | val PCC 0.7467 SG-PCC 0.5760 I-MAE 2.0545 | composite 0.3518 | 107.8s
 >> NEW BEST composite=0.3518 (PCC=0.7467 SG-PCC=0.5760 I-MAE=2.0545)
Ep    11 | lr 1.00e-06 | loss 9.7225 [edm 0.549(2.00) pcc 0.213(7.00) int 1.093(3.00) ssim 0.454(0.50) tail 0.095] | val PCC 0.7502 SG-PCC 0.5791 I-MAE 1.9841 | composite 0.3453 | 108.3s
 >> NEW BEST composite=0.3453 (PCC=0.7502 SG-PCC=0.5791 I-MAE=1.9841)
Ep    16 | lr 1.00e-06 | loss 9.0554 [edm 0.506(2.00) pcc 0.199(7.00) int 1.056(3.00) ssim 0.454(0.50) tail 0.086] | val PCC 0.7601 SG-PCC 0.5943 I-MAE 1.9210 | composite 0.3329 | 107.7s
 >> NEW BEST composite=0.3329 (PCC=0.7601 SG-PCC=0.5943 I-MAE=1.9210)
