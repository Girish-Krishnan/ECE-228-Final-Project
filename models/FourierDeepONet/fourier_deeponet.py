import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VMIN, VMAX = 1500.0, 4500.0

class SeismicDataset(Dataset):
    def __init__(self, pairs):
        branch_lst, target_lst = [], []
        for seis_path, vel_path in pairs:
            seis = np.load(seis_path).astype(np.float32)
            vel  = np.load(vel_path ).astype(np.float32)
            if vel.ndim == 3:
                vel = vel[:, None, :, :]
            branch_lst.append(torch.from_numpy(seis))
            target_lst.append(torch.from_numpy(vel))
        self.branch = torch.cat(branch_lst, dim=0)
        self.target = torch.cat(target_lst, dim=0)
        self.trunk  = torch.zeros(len(self.branch), 1, dtype=torch.float32)
        self.target = (self.target - VMIN) / (VMAX - VMIN) * 2 - 1
        self.target = torch.clamp(self.target, -1.0, 1.0)

    def __len__(self): return len(self.branch)
    def __getitem__(self, idx): return self.branch[idx], self.trunk[idx], self.target[idx]

def collect_pairs(root="."):
    pairs = []
    for fam in ["FlatVel_A", "Style_A"]:
        for sp in glob.glob(os.path.join(root, fam, "data", "*.npy")):
            mp = sp.replace("data", "model")
            if os.path.exists(mp): pairs.append((sp, mp))
    for fam in ["CurveFault_A", "FlatFault_A"]:
        for sp in glob.glob(os.path.join(root, fam, "seis*_*.npy")):
            mp = os.path.join(root, fam, os.path.basename(sp).replace("seis", "vel"))
            if os.path.exists(mp): pairs.append((sp, mp))
    return pairs

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, inp, weights):
        return torch.einsum("bixy,ioxy->boxy", inp, weights)

    def forward(self, x):
        b = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])
        out_ft = torch.zeros(b, self.weights1.shape[1], x.size(-2), x.size(-1)//2+1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfftn(out_ft, s=x.shape[-2:])

class UBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down1 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.down2 = nn.Conv2d(channels, channels, 3, 2, 1)
        self.up1   = nn.ConvTranspose2d(channels, channels, 4, 2, 1)
        self.up2   = nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1)

    def forward(self, x):
        d1 = F.relu(self.down1(x))
        d2 = F.relu(self.down2(d1))
        u1 = F.relu(self.up1(d2))
        u2 = F.relu(self.up2(torch.cat([u1, d1], 1)))
        return u2

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
        b, c, h, w = x.shape
        x1 = self.conv0(x)
        x2 = self.w0(x.view(b, c, -1)).view(b, c, h, w)
        x = F.relu(x1 + x2)
        x1 = self.conv1(x)
        x2 = self.w1(x.view(b, c, -1)).view(b, c, h, w)
        x3 = self.unet(x)
        x = F.relu(x1 + x2 + x3)
        x = F.adaptive_avg_pool2d(x, (70, 70))
        x = F.relu(self.fc1(x.permute(0, 2, 3, 1)))
        return self.fc2(x).permute(0, 3, 1, 2)

class BranchNet(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc0 = nn.Linear(5, width)
    def forward(self, x):
        x = F.pad(x, (1,1,0,0), mode="replicate")
        x = x.permute(0, 3, 2, 1)
        x = self.fc0(x)
        return x.permute(0, 3, 1, 2)

class TrunkNet(nn.Module):
    def __init__(self, width, num_parameter):
        super().__init__()
        self.fc0 = nn.Linear(num_parameter, width)
    def forward(self, x):
        return self.fc0(x)[:, :, None, None]

class FourierDeepONet(nn.Module):
    def __init__(self, num_parameter, width=64, modes1=20, modes2=20):
        super().__init__()
        self.branch = BranchNet(width)
        self.trunk  = TrunkNet(width, num_parameter)
        self.decoder = Decoder(modes1, modes2, width)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        xb = self.branch(inputs[0])
        xt = self.trunk(inputs[1])
        return self.decoder(xb * xt + self.bias)

def denorm(x):
    return x * (VMAX - VMIN) / 2 + (VMAX + VMIN) / 2

def main():
    pairs = collect_pairs(".")
    dataset = SeismicDataset(pairs)

    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.2 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    model = FourierDeepONet(num_parameter=1, width=64, modes1=20, modes2=20).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    for epoch in range(1, 101):
        model.train()
        running = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/100")
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
                unnorm_running += torch.mean(torch.abs(denorm(pred) - denorm(target))).item() * branch.size(0)

        val_mae = val_running / len(val_loader.dataset)
        val_unnorm_mae = unnorm_running / len(val_loader.dataset)
        print(f"Epoch {epoch:2d}  Train MAE {train_mae:.4f}  Val MAE {val_mae:.4f}  Val MAE (m/s) {val_unnorm_mae:.2f}")

        if epoch % 5 == 0:
            torch.save({"model_state_dict": model.state_dict()}, f"fourier_deeponet_epoch{epoch}.pt")

if __name__ == "__main__":
    main()
