
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path

from Dataset import UpscaleDataset
from Network import EDMPrecond

#from torch.utils.data import random_split
from torch.utils.data import Subset


import argparse
import check

parser = argparse.ArgumentParser()
parser.add_argument("--process", required=True, choices=["Train", "Test"])
args = parser.parse_args()


# ============================================================
# DDP SETUP
# ============================================================

def setup_ddp():
    

    if "RANK" not in os.environ:
        print("Running in SINGLE GPU mode (no DDP)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, device

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, torch.device(f"cuda:{rank}")



def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()



# ============================================================
# EDM LOSS (unchanged)
# ============================================================

class DiffusionLoss:
    def __init__(self, sigma_min=0.002, sigma_max=40, sigma_data=1):   # earlier 40 , 
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def sample_sigma(self, B, device):
        u = torch.rand(B, device=device)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** u

    def __call__(self, model, residual, coarse_up, labels):

        B = residual.shape[0]
        device = residual.device

        sigma = self.sample_sigma(B, device)

        noise = torch.randn_like(residual)
        noisy = residual + sigma.view(-1,1,1,1) * noise

        sigma2 = sigma**2
        sd = self.sigma_data

        c_skip = (sd**2)/(sigma2+sd**2)
        c_out  = sigma*sd/torch.sqrt(sigma2+sd**2)
        c_in   = 1/torch.sqrt(sigma2+sd**2)

        c_skip = c_skip.view(-1,1,1,1)
        c_out  = c_out.view(-1,1,1,1)
        c_in   = c_in.view(-1,1,1,1)

        F_theta = model(noisy*c_in, sigma, condition_img=coarse_up, class_labels=labels)

        x_hat = c_skip*noisy + c_out*F_theta

        return F.mse_loss(x_hat, residual)


# ============================================================
# TRAIN EPOCH
# ============================================================

def train_epoch(model, loader, loss_fn, opt, device):

    model.train()
    total = 0

    for batch in loader:

        residual = batch["targets"].to(device, non_blocking=True)
        coarse   = batch["inputs"].to(device, non_blocking=True)
        labels   = torch.stack([batch["doy"], batch["hour"]],1).to(device)

        loss = loss_fn(model, residual, coarse, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    return total/len(loader)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0

    for batch in loader:
        residual = batch["targets"].to(device)
        coarse   = batch["inputs"].to(device)
        labels   = torch.stack([batch["doy"], batch["hour"]],1).to(device)

        loss = loss_fn(model, residual, coarse, labels)
        total += loss.item()

    return total/len(loader)

# ============================================================
# MAIN
# ============================================================

def main():

    rank, device = setup_ddp()

    # -------------------------
    # Dataset split
    # -------------------------
    full_dataset = UpscaleDataset(
        "data/RF_1975to2023.nc",
        downscale_factor=4,
        normalize=True
    )

    N = len(full_dataset)
    
    train_end = int(0.8 * N)
    val_end   = int(0.9 * N)
    
    train_set = Subset(full_dataset, range(0, train_end))
    val_set   = Subset(full_dataset, range(train_end, val_end))
    test_set  = Subset(full_dataset, range(val_end, N))


 

    # -------------------------
    # Model
    # -------------------------
    model = EDMPrecond(
        img_resolution=(full_dataset.H, full_dataset.W),
        in_channels=2,
        out_channels=1,
        label_dim=2,
        sigma_data=1.0
    ).to(device)

    if args.process == "Train":
        model = DDP(model, device_ids=[rank])

    loss_fn = DiffusionLoss()

    # =========================================================
    # TRAIN MODE
    # =========================================================
    if args.process == "Train":

        train_sampler = DistributedSampler(train_set)

        train_loader = DataLoader(
            train_set, batch_size=64, sampler=train_sampler,
            num_workers=4, pin_memory=True, persistent_workers=True
        )

        val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

        opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

        best_loss = 1e9
        patience = 0

        for epoch in range(3000):

            train_sampler.set_epoch(epoch)

            train_loss = train_epoch(model, train_loader, loss_fn, opt, device)
            val_loss   = eval_epoch(model, val_loader, loss_fn, device)

            if rank == 0:
                print(f"Epoch {epoch:04d} | train {train_loss:.5f} | val {val_loss:.5f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience = 0
                    torch.save(model.module.state_dict(), "checkpoints/best_rerun.pt")
                else:
                    patience += 1

                if patience >= 300:
                    print("Early stopping triggered")
                    break

    # =========================================================
    # TEST MODE
    # =========================================================
    elif args.process == "Test":

      if rank == 0:
          print("Loading checkpoint...")
  
      # ALWAYS define target_model FIRST
      target_model = model.module if hasattr(model, "module") else model
  
      ckpt = torch.load("checkpoints/best.pt", map_location=device)
  
      target_model.load_state_dict(ckpt["model_state_dict"])
  
      test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
  
      test_loss = eval_epoch(target_model, test_loader, loss_fn, device)
  
      if rank == 0:
          print(f"TEST loss: {test_loss:.5f}")
  
          # pass SAME target_model
          check.main(target_model, full_dataset, test_set, device)




    cleanup_ddp()


    
if __name__ == "__main__":
    main()
