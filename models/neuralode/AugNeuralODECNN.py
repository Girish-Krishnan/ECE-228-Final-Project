import os, glob, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

BATCH          = 8
EPOCHS         = 150
FM_CH          = 192
VEC_CH         = 384
AUG_FM_CH      = 32
AUG_VEC_CH     = 64
LR             = 3e-4
WD             = 5e-5
CLIP           = 1.0
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
AMP            = torch.cuda.is_available()

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class OpenFWISubset(Dataset):
    def __init__(self, root="."):
        pair_list = []
        for fam in ["FlatVel_A", "Style_A"]:
            for sp in glob.glob(os.path.join(root, fam, "data", "*.npy")):
                mp = sp.replace("data", "model")
                if os.path.exists(mp):
                    pair_list.append((sp, mp))
        for fam in ["CurveFault_A", "FlatFault_A"]:
            for sp in glob.glob(os.path.join(root, fam, "seis*_*.npy")):
                mp = os.path.join(root, fam, os.path.basename(sp).replace("seis", "vel"))
                if os.path.exists(mp):
                    pair_list.append((sp, mp))

        seis_list, vel_list = [], []
        for sp, mp in pair_list:
            seis = np.load(sp).astype(np.float32)                    # N,5,1000,70
            vel  = np.load(mp).astype(np.float32)                    # N,70,70 or N,1,70,70
            if vel.ndim == 3:
                vel = vel[:, None]
            seis_list.append(torch.from_numpy(seis))
            vel_list.append(torch.from_numpy(vel))

        self.seis = torch.cat(seis_list, 0)
        self.vel  = torch.cat(vel_list , 0)

        # amplitude log scale normalisation
        amp = torch.clamp(self.seis, -60, 60)
        self.seis = torch.log1p(torch.abs(amp)) * torch.sign(amp) / math.log(61)

        # velocity min–max to [-1,1]
        self.vmin, self.vmax = 1500.0, 4500.0
        self.vel = (self.vel - self.vmin) / (self.vmax - self.vmin) * 2 - 1

    def __len__(self):
        return len(self.seis)

    def __getitem__(self, idx):
        return self.seis[idx], self.vel[idx]

def conv_gn_relu(inp, outp, k=3, s=1, p=1, groups=8):
    return nn.Sequential(
        nn.Conv2d(inp, outp, k, s, p, bias=False),
        nn.GroupNorm(groups, outp),
        nn.GELU())

class SE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // r, 1)
        self.fc2 = nn.Conv2d(ch // r, ch, 1)
    def forward(self, x):
        w = F.gelu(self.fc1(F.adaptive_avg_pool2d(x, 1)))
        w = torch.sigmoid(self.fc2(w))
        return x * w

class ResUnit(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            conv_gn_relu(ch, ch),
            conv_gn_relu(ch, ch),
            SE(ch))
    def forward(self, x): return self.block(x) + x

class ODEFm(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            conv_gn_relu(ch, ch),
            conv_gn_relu(ch, ch),
            nn.Conv2d(ch, ch, 1))
    def forward(self, t, h): return self.net(h)

class ODEVec(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim))
    def forward(self, t, h): return self.net(h)

def integrate(h0, func, n_steps=4):
    assert h0.dtype == torch.float32, "keep everything float32 on MPS"
    device = h0.device
    t = torch.linspace(0, 1, n_steps + 1, device=device,
                       dtype=torch.float32)          # 0,0.25,…,1

    out = odeint(func, h0, t, method="rk4")          # (T,B,…)
    return out[-1]                                   # state at t=1

class AugNODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem_time = nn.Conv1d(5, 32, kernel_size=21, stride=4, padding=10)  # 1000→250
        # spatial encoder
        self.enc2d = nn.Sequential(
            conv_gn_relu(32, 64, 3, 2, 1), ResUnit(64),
            conv_gn_relu(64, 128, 3, 2, 1), ResUnit(128),
            conv_gn_relu(128, FM_CH, 3, 2, 1), ResUnit(FM_CH))
        # ODE on feature map
        self.fm_ode_func = ODEFm(FM_CH + AUG_FM_CH)
        # projection to vector
        self.fm_to_vec = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(FM_CH, VEC_CH))
        # ODE on vector
        self.vec_ode_func = ODEVec(VEC_CH + AUG_VEC_CH)
        # decoder
        self.dec = nn.Sequential(
            nn.Linear(VEC_CH, VEC_CH * 2), nn.GELU(),
            # ↓  4 · 35 · 35 instead of 16 · 35 · 35
            nn.Linear(VEC_CH * 2, 4 * 35 * 35)
        )
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        # x: B,5,1000,70
        b = x.size(0)
        x = self.stem_time(x.view(b, 5, 1000 * 70))       # B,32,250*70
        x = x.view(b, 32, 250, 70)                        # reshape
        fm = self.enc2d(x)                                # B,C,H,W
        fm_aug = torch.cat([fm, torch.zeros(b, AUG_FM_CH, fm.size(2), fm.size(3), device=x.device)], 1)
        fm_out = integrate(fm_aug, self.fm_ode_func)[:, :FM_CH]
        vec = self.fm_to_vec(fm_out)                      # B,VEC_CH
        vec_aug = torch.cat([vec, torch.zeros(b, AUG_VEC_CH, device=x.device)], 1)
        vec_out = integrate(vec_aug, self.vec_ode_func)[:, :VEC_CH]
        
        # after the second ODE stage …
        dec = self.dec(vec_out)         # (B, 4·35·35)
        b = dec.size(0)
        dec = dec.view(b, 4, 35, 35)  
        out = self.up(dec)              # PixelShuffle(2) → (B, 1, 70, 70)
        return out.tanh()


def vel_denorm(t):
    return (t + 1) / 2 * (4500.0 - 1500.0) + 1500.0

def run():
    ds = OpenFWISubset(".")
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train],
                                    generator=torch.Generator().manual_seed(0))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds  , batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)

    model = AugNODE().to(DEVICE)
    ema   = torch.optim.swa_utils.AveragedModel(model)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    l1 = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_dl, leave=False, desc=f"Epoch {epoch}/{EPOCHS}")
        for seis, vel in pbar:
            seis, vel = seis.to(DEVICE), vel.to(DEVICE)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=AMP):
                pred = model(seis)
                loss_l1 = l1(pred, vel)
                loss_ss = 1 - ssim_fn((pred + 1) / 2, (vel + 1) / 2, data_range=1.0)
                loss = 0.85 * loss_l1 + 0.15 * loss_ss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt)
            scaler.update()
            ema.update_parameters(model)
            running += loss_l1.item() * seis.size(0)
            pbar.set_postfix(l1=loss_l1.item())
        sched.step()
        train_mae = running / len(train_dl.dataset)

        # validation
        model.eval(); ema.eval()
        val_l1_sum, val_denorm_sum = 0.0, 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=AMP):
            for seis, vel in val_dl:
                seis, vel = seis.to(DEVICE), vel.to(DEVICE)
                pred = ema(seis)
                val_l1_sum += l1(pred, vel).item() * seis.size(0)
                val_denorm_sum += torch.mean(torch.abs(
                    vel_denorm(pred) - vel_denorm(vel))).item() * seis.size(0)
        val_mae = val_l1_sum / len(val_dl.dataset)
        val_mae_denorm = val_denorm_sum / len(val_dl.dataset)
        print(f"Epoch {epoch:3d}  Train MAE {train_mae:.4f}  "
              f"Val MAE {val_mae:.4f}  Val MAE (denorm) {val_mae_denorm:.2f} m/s")

        if epoch % 10 == 0:
            torch.save({"model": model.state_dict(),
                        "ema":   ema.state_dict()},
                       f"aug_node_plus_epoch{epoch}.pth")

    import matplotlib.pyplot as plt

    # Visualize predictions
    model.eval()
    ema.eval()
    seis, vel = next(iter(val_dl))
    seis, vel = seis.to(DEVICE), vel.to(DEVICE)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=AMP):
        pred = ema(seis)

    # Denormalize velocity for visualization
    vel_denorm_pred = vel_denorm(pred).cpu().numpy()
    vel_denorm_true = vel_denorm(vel).cpu().numpy()

    for i in range(min(5, len(seis))):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(vel_denorm_true[i, 0], cmap='jet', aspect='auto')
        ax[0].set_title('Ground Truth Velocity (m/s)')
        ax[0].axis('off')

        ax[1].imshow(vel_denorm_pred[i, 0], cmap='jet', aspect='auto')
        ax[1].set_title('Predicted Velocity (m/s)')
        ax[1].axis('off')

        plt.show()


if __name__ == "__main__":
    run()
