import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SeismicDataset(Dataset):
    """
    Each __getitem__ returns:
        branch_data : (5, 1000, 70)   seismic cube    -> used by Branch net
        trunk_data  : (1,)            dummy feature   -> used by Trunk  net
        target      : (1, 70, 70)     velocity map
    """

    def __init__(self, pairs):
        branch_lst, target_lst = [], []
        for seis_path, vel_path in pairs:
            seis = np.load(seis_path).astype(np.float32)     # (N, 5, 1000, 70)
            vel  = np.load(vel_path ).astype(np.float32)     # (N, 1, 70, 70) or (N, 70, 70)
            if vel.ndim == 3:
                vel = vel[:, None, :, :]
            branch_lst.append(torch.from_numpy(seis))
            target_lst.append(torch.from_numpy(vel))
        self.branch = torch.cat(branch_lst, dim=0)
        self.target = torch.cat(target_lst, dim=0)
        # dummy trunk vector of zeros (one scalar) for each sample
        self.trunk  = torch.zeros(len(self.branch), 1, dtype=torch.float32)

        # simple min-max to [-1,1]
        self.target = (self.target - 1500.0) / 1500.0    # roughly within [-1,2]
        self.target = torch.clamp(self.target, -1.0, 1.0)

    def __len__(self):
        return len(self.branch)

    def __getitem__(self, idx):
        return self.branch[idx], self.trunk[idx], self.target[idx]

def collect_pairs(root="."):
    pairs = []
    # Vel & Style
    for fam in ["FlatVel_A", "Style_A"]:
        for sp in glob.glob(os.path.join(root, fam, "data", "*.npy")):
            mp = sp.replace("data", "model")
            if os.path.exists(mp):
                pairs.append((sp, mp))
    # Fault
    for fam in ["CurveFault_A", "FlatFault_A"]:
        for sp in glob.glob(os.path.join(root, fam, "seis*_*.npy")):
            mp = os.path.join(root, fam, os.path.basename(sp).replace("seis", "vel"))
            if os.path.exists(mp):
                pairs.append((sp, mp))
    return pairs

# ----------------------------- model -------------------------------- #
# ─── in fourier_deeponet_core.py ─────────────────────────────
class SpectralConv2d(nn.Module):
    """
    Same math, but weights are stored as two real tensors so that DDP/NCCL/Gloo
    can broadcast & allreduce them.
    """
    def __init__(self, c_in, c_out, modes1, modes2):
        super().__init__()
        self.m1, self.m2 = modes1, modes2
        scale = 1.0 / (c_in * c_out)

        # real & imag parts – plain float32 parameters
        self.w1_r = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2))
        self.w1_i = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2))
        self.w2_r = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2))
        self.w2_i = nn.Parameter(scale * torch.randn(c_in, c_out, modes1, modes2))

    @staticmethod
    def _to_complex(w_r, w_i):
        return torch.view_as_complex(torch.stack((w_r, w_i), dim=-1))

    def compl_mul2d(self, x, w_r, w_i):
        w = self._to_complex(w_r, w_i)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))
        out_ft = torch.zeros(
            B, self.w1_r.size(1), H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[..., :self.m1, :self.m2] = self.compl_mul2d(
            x_ft[..., :self.m1, :self.m2], self.w1_r, self.w1_i
        )
        out_ft[..., -self.m1:, :self.m2] = self.compl_mul2d(
            x_ft[..., -self.m1:, :self.m2], self.w2_r, self.w2_i
        )

        return torch.fft.irfftn(out_ft, s=(H, W))

class UBlock(nn.Module):
    """mini U-Net block, returns width channels"""
    def __init__(self, channels):
        super().__init__()
        self.down1 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down2 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.up1   = nn.ConvTranspose2d(channels, channels, 4, 2, 1)
        self.up2   = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)

    def forward(self, x):
        d1 = F.relu(self.down1(x))               # channels
        d2 = F.relu(self.down2(d1))              # channels
        u1 = F.relu(self.up1(d2))                # channels
        u2 = F.relu(self.up2(torch.cat([u1, d1], 1)))  # channels
        return u2                                # <- **keep width channels**

class Decoder(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.unet = UBlock(width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        b, c, h, w = x.shape               # w is 1000
        x1 = self.conv0(x)
        x2 = self.w0(x.view(b, c, -1)).view(b, c, h, w)
        x  = F.relu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(b, c, -1)).view(b, c, h, w)
        x3 = self.unet(x)
        x  = F.relu(x1 + x2 + x3)

        # ---------- new line: shrink to 70 × 70 ----------
        x = F.adaptive_avg_pool2d(x, (70, 70))
        # --------------------------------------------------

        x = x.permute(0, 2, 3, 1)          # b 70 70 64
        x = F.relu(self.fc1(x))            # b 70 70 128
        x = self.fc2(x)                    # b 70 70 1
        x = x.permute(0, 3, 1, 2)          # b 1 70 70
        return x

class BranchNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc0 = nn.Linear(5, width)
    def forward(self, x):
        # x: (b, 5, 1000, 70)
        x = F.pad(x, (1,1,0,0), mode="replicate")     # -> (b,5,1000,72)
        x = x.permute(0, 3, 2, 1)                     # (b,72,1000,5)
        x = self.fc0(x)                               # (b,72,1000,width)
        return x.permute(0, 3, 1, 2)                  # (b,width,72,1000)

class TrunkNet(nn.Module):
    def __init__(self, width, num_parameter):
        super().__init__()
        self.fc0 = nn.Linear(num_parameter, width)
    def forward(self, x):
        x = self.fc0(x)
        return x[:, :, None, None]                    # (b,width,1,1)

class FourierDeepONet(nn.Module):
    def __init__(self, num_parameter, width=64, modes1=20, modes2=20):
        super().__init__()
        self.branch = BranchNet(width)
        self.trunk  = TrunkNet(width, num_parameter)
        self.decoder = Decoder(modes1, modes2, width)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        branch_in, trunk_in = inputs
        xb = self.branch(branch_in)
        xt = self.trunk(trunk_in)
        x  = xb * xt + self.bias
        return self.decoder(x)

# ----------------------------- training ----------------------------- #
def denorm(x):
    return x * 1500.0 + 1500.0   # x is a tensor on the same device

def main():
    pairs = collect_pairs(".")
    dataset = SeismicDataset(pairs)

    train_len = int(0.8 * len(dataset))
    val_len   = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model = FourierDeepONet(num_parameter=1, width=64, modes1=20, modes2=20).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    epochs = 100
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for branch, trunk, target in loop:
            branch, trunk, target = branch.to(DEVICE), trunk.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            pred = model((branch, trunk))
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            running += loss.item() * branch.size(0)
            loop.set_postfix(train_mae=loss.item())
        train_mae = running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        unnorm_running = 0.0
        with torch.no_grad():
            for branch, trunk, target in val_loader:
                branch, trunk, target = branch.to(DEVICE), trunk.to(DEVICE), target.to(DEVICE)
                pred = model((branch, trunk))
                val_running += criterion(pred, target).item() * branch.size(0)

                pred_real   = denorm(pred)
                target_real = denorm(target)
                unnorm_mae  = torch.mean(torch.abs(pred_real - target_real)).item()
                unnorm_running += unnorm_mae * branch.size(0)

        val_mae = val_running / len(val_loader.dataset)
        val_unnorm_mae = unnorm_running / len(val_loader.dataset)
        print(f"Epoch {epoch:2d}  Train MAE {train_mae:.4f}  Val MAE (Normalized) {val_mae:.4f} Val MAE (Unnormalized) {val_unnorm_mae:.4f}")

        # optional checkpoint
        if epoch % 5 == 0:
            torch.save({"model_state_dict": model.state_dict()},
                       f"fourier_deeponet_epoch{epoch}.pt")

if __name__ == "__main__":
    main()