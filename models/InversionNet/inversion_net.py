#!/usr/bin/env python
import argparse, glob, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from math import ceil
from tqdm import tqdm

from pytorch_msssim import ssim as torch_ssim

class SmallOpenFWI(Dataset):
    def __init__(self, pair_list):
        b, t = [], []
        for seis, vel in pair_list:
            s = np.load(seis).astype(np.float32)
            v = np.load(vel ).astype(np.float32)
            if v.ndim == 3: v = v[:, None, :, :]
            b.append(torch.from_numpy(s))
            t.append(torch.from_numpy(v))
        self.seis = torch.cat(b, 0)
        self.vel  = torch.cat(t, 0)
        self.vel  = torch.clamp((self.vel - 1500.0) / 1500.0, -1.0, 1.0)

    def __len__(self): return len(self.seis)
    def __getitem__(self, i): return self.seis[i], self.vel[i]

def gather_pairs(root):
    pairs=[]
    for fam in ["FlatVel_A","Style_A"]:
        for sp in glob.glob(os.path.join(root, fam, "data", "*.npy")):
            mp = sp.replace("data","model")
            if os.path.exists(mp): pairs.append((sp, mp))
    for fam in ["CurveFault_A","FlatFault_A"]:
        for sp in glob.glob(os.path.join(root, fam, "seis*_*.npy")):
            mp = os.path.join(root, fam, os.path.basename(sp).replace("seis","vel"))
            if os.path.exists(mp): pairs.append((sp, mp))
    return pairs

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

class ConvBlock(nn.Module):
    def __init__(self,in_f,out_f,kernel_size=3,stride=1,pad=1,norm='bn',relu_s=0.2):
        super().__init__()
        layers=[nn.Conv2d(in_f,out_f,kernel_size,stride,pad)]
        if norm in NORM_LAYERS: layers.append(NORM_LAYERS[norm](out_f))
        layers.append(nn.LeakyReLU(relu_s,True))
        self.layers=nn.Sequential(*layers)
    def forward(self,x): return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self,in_f,out_f,kernel_size=3,stride=1,pad=1,norm='bn'):
        super().__init__()
        layers=[nn.Conv2d(in_f,out_f,kernel_size,stride,pad)]
        if norm in NORM_LAYERS: layers.append(NORM_LAYERS[norm](out_f))
        layers.append(nn.Tanh())
        self.layers=nn.Sequential(*layers)
    def forward(self,x): return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self,in_f,out_f,ks=2,stride=2,pad=0,norm='bn'):
        super().__init__()
        layers=[nn.ConvTranspose2d(in_f,out_f,ks,stride,pad)]
        if norm in NORM_LAYERS: layers.append(NORM_LAYERS[norm](out_f))
        layers.append(nn.LeakyReLU(0.2,True))
        self.layers=nn.Sequential(*layers)
    def forward(self,x): return self.layers(x)

class InversionNet(nn.Module):
    def __init__(self,d1=32,d2=64,d3=128,d4=256,d5=512,ratio=1.0):
        super().__init__()
        self.c1 = ConvBlock(5,d1,kernel_size=(7,1),stride=(2,1),pad=(3,0))
        self.c2_1=ConvBlock(d1,d2,kernel_size=(3,1),stride=(2,1),pad=(1,0)); self.c2_2=ConvBlock(d2,d2,kernel_size=(3,1),pad=(1,0))
        self.c3_1=ConvBlock(d2,d2,kernel_size=(3,1),stride=(2,1),pad=(1,0)); self.c3_2=ConvBlock(d2,d2,kernel_size=(3,1),pad=(1,0))
        self.c4_1=ConvBlock(d2,d3,kernel_size=(3,1),stride=(2,1),pad=(1,0)); self.c4_2=ConvBlock(d3,d3,kernel_size=(3,1),pad=(1,0))
        self.c5_1=ConvBlock(d3,d3,stride=2); self.c5_2=ConvBlock(d3,d3)
        self.c6_1=ConvBlock(d3,d4,stride=2); self.c6_2=ConvBlock(d4,d4)
        self.c7_1=ConvBlock(d4,d4,stride=2); self.c7_2=ConvBlock(d4,d4)
        self.c8   = ConvBlock(d4,d5,kernel_size=(8,ceil(70*ratio/8)),pad=0)
        self.d1_1=DeconvBlock(d5,d5,ks=5);           self.d1_2=ConvBlock(d5,d5)
        self.d2_1=DeconvBlock(d5,d4,ks=4,pad=1);     self.d2_2=ConvBlock(d4,d4)
        self.d3_1=DeconvBlock(d4,d3,ks=4,pad=1);     self.d3_2=ConvBlock(d3,d3)
        self.d4_1=DeconvBlock(d3,d2,ks=4,pad=1);     self.d4_2=ConvBlock(d2,d2)
        self.d5_1=DeconvBlock(d2,d1,ks=4,pad=1);     self.d5_2=ConvBlock(d1,d1)
        self.out = ConvBlock_Tanh(d1,1)

    def forward(self,x):
        x=self.c1(x)
        x=self.c2_1(x); x=self.c2_2(x)
        x=self.c3_1(x); x=self.c3_2(x)
        x=self.c4_1(x); x=self.c4_2(x)
        x=self.c5_1(x); x=self.c5_2(x)
        x=self.c6_1(x); x=self.c6_2(x)
        x=self.c7_1(x); x=self.c7_2(x)
        x=self.c8(x)
        x=self.d1_1(x); x=self.d1_2(x)
        x=self.d2_1(x); x=self.d2_2(x)
        x=self.d3_1(x); x=self.d3_2(x)
        x=self.d4_1(x); x=self.d4_2(x)
        x=self.d5_1(x); x=self.d5_2(x)
        x=F.pad(x,[-5,-5,-5,-5])
        return self.out(x)

def denorm(t): return t*1500.0 + 1500.0

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    pairs  = gather_pairs(args.root)
    data   = SmallOpenFWI(pairs)

    n = len(data)
    n_train = int(0.7 * n)
    n_val   = int(0.2 * n)
    n_test  = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(data, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

    model = InversionNet().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train(); tr = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for s, v in loop:
            s, v = s.to(device), v.to(device)
            opt.zero_grad()
            p = model(s)
            loss = crit(p, v)
            loss.backward(); opt.step()
            tr += loss.item() * s.size(0)
            loop.set_postfix(mae=loss.item())
        tr /= len(train_loader.dataset)

        model.eval(); vl = 0.0; vl_real = 0.0
        with torch.no_grad():
            for s, v in val_loader:
                s, v = s.to(device), v.to(device)
                p = model(s)
                vl += crit(p, v).item() * s.size(0)
                vl_real += torch.mean(torch.abs(denorm(p) - denorm(v))).item() * s.size(0)
        vl /= len(val_loader.dataset)
        vl_real /= len(val_loader.dataset)
        print(f"Epoch {epoch:2d}  Train MAE {tr:.4f}  Val MAE(norm) {vl:.4f}  Val MAE(real) {vl_real:.2f} m/s")

        if epoch % 5 == 0 and args.ckpt:
            torch.save({'model_state_dict': model.state_dict()},
                       f"{args.ckpt}_{epoch}.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data", help="root folder with dataset families")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--ckpt", type=str, default="inversionnet_epoch", help="checkpoint prefix ('' to disable)")
    args = ap.parse_args()
    run(args)
