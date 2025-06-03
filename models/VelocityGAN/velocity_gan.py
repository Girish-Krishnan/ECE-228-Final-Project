#!/usr/bin/env python
"""
train_velocitygan.py
--------------------
Minimal Velocity-GAN on the small OpenFWI/Kaggle subset.

Generator  : LANL InversionNet (maps 5×1000×70 seismic cube → 1×70×70 velocity)
Discriminator : LANL Patch discriminator

Loss
----
* Discriminator :  ½·(BCE(real, 1) + BCE(fake, 0))
* Generator     :  BCE(fake, 1)   +  λ·L1(fake, target)   (λ = 100)

The L1 term is what Kaggle scores (MAE).

Run
----
python train_velocitygan.py \
       --data_root ./data \
       --epochs 25 \
       --batch 8
"""

import argparse, glob, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from math import ceil
from tqdm import tqdm
from pytorch_msssim import ssim

# ------------------------------------------------ Dataset ------------------------------------------------ #
class SmallOpenFWI(Dataset):
    def __init__(self, pairs):
        seis, vel = [], []
        for sp, vp in pairs:
            s = np.load(sp).astype(np.float32)               # (N,5,1000,70)
            v = np.load(vp).astype(np.float32)               # (N,1,70,70) or (N,70,70)
            if v.ndim == 3: v = v[:, None, :, :]
            seis.append(torch.from_numpy(s))
            vel .append(torch.from_numpy(v))
        self.seis = torch.cat(seis, 0)
        self.vel  = torch.clamp((torch.cat(vel,0) - 1500.0) / 1500.0, -1, 1)   # → [-1,1]

    def __len__(self): return len(self.seis)
    def __getitem__(self, i): return self.seis[i], self.vel[i]

def gather_pairs(root):
    pairs=[]
    for fam in ["FlatVel_A","Style_A"]:
        for sp in glob.glob(os.path.join(root,fam,"data","*.npy")):
            mp = sp.replace("data","model")
            if os.path.exists(mp): pairs.append((sp, mp))
    for fam in ["CurveFault_A","FlatFault_A"]:
        for sp in glob.glob(os.path.join(root,fam,"seis*_*.npy")):
            mp=os.path.join(root,fam,os.path.basename(sp).replace("seis","vel"))
            if os.path.exists(mp): pairs.append((sp, mp))
    return pairs


# -------------------------------------- building blocks -------------------------------------------------- #
NORM = { 'bn': nn.BatchNorm2d }

class ConvB(nn.Module):
    def __init__(self,i,o,k=3,s=1,p=1): super().__init__()
    def __init__(self,i,o,k=3,s=1,p=1,n='bn'):
        super().__init__()
        layers=[nn.Conv2d(i,o,k,s,p)]
        if n in NORM: layers.append(NORM[n](o))
        layers.append(nn.LeakyReLU(0.2,True))
        self.seq=nn.Sequential(*layers)
    def forward(self,x): return self.seq(x)

class ConvB_Tanh(nn.Module):
    def __init__(self,i,o,k=3,s=1,p=1,n='bn'):
        super().__init__()
        layers=[nn.Conv2d(i,o,k,s,p)]
        if n in NORM: layers.append(NORM[n](o))
        layers.append(nn.Tanh())
        self.seq=nn.Sequential(*layers)
    def forward(self,x): return self.seq(x)

class DeconvB(nn.Module):
    def __init__(self,i,o,k=2,s=2,p=0,n='bn'):
        super().__init__()
        layers=[nn.ConvTranspose2d(i,o,k,s,p)]
        if n in NORM: layers.append(NORM[n](o))
        layers.append(nn.LeakyReLU(0.2,True))
        self.seq=nn.Sequential(*layers)
    def forward(self,x): return self.seq(x)

# ----------------------- Generator: InversionNet --------------------- #
class InversionNet(nn.Module):
    def __init__(self,d1=32,d2=64,d3=128,d4=256,d5=512,ratio=1.0):
        super().__init__()
        self.c1 = ConvB(5,d1,k=(7,1),s=(2,1),p=(3,0))
        self.c2_1=ConvB(d1,d2,k=(3,1),s=(2,1),p=(1,0)); self.c2_2=ConvB(d2,d2,k=(3,1),p=(1,0))
        self.c3_1=ConvB(d2,d2,k=(3,1),s=(2,1),p=(1,0)); self.c3_2=ConvB(d2,d2,k=(3,1),p=(1,0))
        self.c4_1=ConvB(d2,d3,k=(3,1),s=(2,1),p=(1,0)); self.c4_2=ConvB(d3,d3,k=(3,1),p=(1,0))
        self.c5_1=ConvB(d3,d3,s=2); self.c5_2=ConvB(d3,d3)
        self.c6_1=ConvB(d3,d4,s=2); self.c6_2=ConvB(d4,d4)
        self.c7_1=ConvB(d4,d4,s=2); self.c7_2=ConvB(d4,d4)
        self.c8   = ConvB(d4,d5,k=(8,ceil(70*ratio/8)),p=0)
        self.d1_1=DeconvB(d5,d5,k=5);           self.d1_2=ConvB(d5,d5)
        self.d2_1=DeconvB(d5,d4,k=4,p=1);       self.d2_2=ConvB(d4,d4)
        self.d3_1=DeconvB(d4,d3,k=4,p=1);       self.d3_2=ConvB(d3,d3)
        self.d4_1=DeconvB(d3,d2,k=4,p=1);       self.d4_2=ConvB(d2,d2)
        self.d5_1=DeconvB(d2,d1,k=4,p=1);       self.d5_2=ConvB(d1,d1)
        self.out = ConvB_Tanh(d1,1)
    def forward(self,x):
        x=self.c1(x); x=self.c2_1(x); x=self.c2_2(x)
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
        return self.out(x)               # tanh -> [-1,1]

# ---------------------- Discriminator -------------------------------- #
class Discriminator(nn.Module):
    def __init__(self,d1=32,d2=64,d3=128,d4=256):
        super().__init__()
        self.b1_1=ConvB(1,d1,s=2);  self.b1_2=ConvB(d1,d1)
        self.b2_1=ConvB(d1,d2,s=2); self.b2_2=ConvB(d2,d2)
        self.b3_1=ConvB(d2,d3,s=2); self.b3_2=ConvB(d3,d3)
        self.b4_1=ConvB(d3,d4,s=2); self.b4_2=ConvB(d4,d4)
        self.b5   = ConvB(d4,1,k=5,p=0)
    def forward(self,x):
        x=self.b1_1(x); x=self.b1_2(x)
        x=self.b2_1(x); x=self.b2_2(x)
        x=self.b3_1(x); x=self.b3_2(x)
        x=self.b4_1(x); x=self.b4_2(x)
        x=self.b5(x)
        return x.view(x.size(0),-1)    # (B,1)

# ------------------------------------- helpers ------------------------------------ #
def denorm(t): return t*1500.0 + 1500.0

# ------------------------------------ training loop -------------------------------- #
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    pairs  = gather_pairs(args.data_root)
    ds     = SmallOpenFWI(pairs)

    tr_len = int(0.8*len(ds))
    tr_ds, val_ds = random_split(ds,[tr_len, len(ds)-tr_len])
    tr_loader = DataLoader(tr_ds,batch_size=args.batch,shuffle=True,num_workers=2,pin_memory=True)
    va_loader = DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=2,pin_memory=True)

    G = InversionNet().to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))

    bce   = nn.BCEWithLogitsLoss()
    l1    = nn.L1Loss()
    lam_l1 = 100.0                     # weight for MAE term

    for epoch in range(1, args.epochs+1):
        G.train(); D.train()
        d_running, g_running = 0.0, 0.0
        loop=tqdm(tr_loader,desc=f"Epoch {epoch}/{args.epochs}")
        for s, v in loop:
            s, v = s.to(device), v.to(device)
            real_lbl = torch.ones((s.size(0),1),device=device)
            fake_lbl = torch.zeros_like(real_lbl)

            # ----------- 1. update Discriminator -----------
            with torch.no_grad():
                fake = G(s)
            d_real = D(v)
            d_fake = D(fake.detach())
            loss_D = 0.5*(bce(d_real,real_lbl)+bce(d_fake,fake_lbl))
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # ----------- 2. update Generator ---------------
            fake = G(s)
            d_fake = D(fake)
            adv_loss = bce(d_fake, real_lbl)
            l1_loss  = l1(fake, v)
            loss_G   = adv_loss + lam_l1 * l1_loss
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            d_running += loss_D.item()*s.size(0)
            g_running += l1_loss.item()*s.size(0)         # MAE for monitor
            loop.set_postfix(D=loss_D.item(), G_L1=l1_loss.item())

        mae_train = g_running / len(tr_loader.dataset)

        # ------------------- validation ------------------
        G.eval(); val_mae_n, val_mae_real = 0.0, 0.0
        with torch.no_grad():
            for s, v in va_loader:
                s,v = s.to(device), v.to(device)
                p = G(s)
                val_mae_n   += l1(p,v).item()*s.size(0)
                val_mae_real+= torch.mean(torch.abs(denorm(p)-denorm(v))).item()*s.size(0)
        val_mae_n   /= len(va_loader.dataset)
        val_mae_real/= len(va_loader.dataset)
        print(f"Epoch {epoch:2d} | Train MAE {mae_train:.4f} | Val MAE(norm) {val_mae_n:.4f} | Val MAE(real) {val_mae_real:.2f} m/s")

        if epoch % 5 == 0:
            torch.save({'G':G.state_dict(),'D':D.state_dict()},
                       f"velocitygan_epoch{epoch}.pt")

# ---------------------------------- main ------------------------------------------ #
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",default="./",help="root with dataset families")
    ap.add_argument("--epochs",type=int,default=50)
    ap.add_argument("--batch",type=int,default=8)
    args=ap.parse_args()
    train(args)
