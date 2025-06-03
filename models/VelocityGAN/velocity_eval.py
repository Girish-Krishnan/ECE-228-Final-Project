#!/usr/bin/env python
"""
vis_velocitygan.py
------------------
Load a trained Velocity-GAN checkpoint (generator + discriminator state-dict)
and display a few random predictions versus the true velocity maps.

Example
-------
python vis_velocitygan.py \
        --ckpt velocitygan_epoch25.pt \
        --data_root ./data \
        --num 3
"""

import argparse, glob, os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from math import ceil

# ------------------------------------- data -------------------------------------- #
class SmallOpenFWI(torch.utils.data.Dataset):
    def __init__(self, pairs):
        s_lst, v_lst = [], []
        for sp, vp in pairs:
            s = np.load(sp).astype(np.float32)               # (N,5,1000,70)
            v = np.load(vp).astype(np.float32)               # (N,1,70,70) or (N,70,70)
            if v.ndim == 3: v = v[:, None, :, :]
            s_lst.append(torch.from_numpy(s))
            v_lst.append(torch.from_numpy(v))
        self.seis = torch.cat(s_lst, 0)
        self.vel  = torch.clamp((torch.cat(v_lst, 0) - 1500.0) / 1500.0, -1, 1)

    def __len__(self): return len(self.seis)
    def __getitem__(self, i): return self.seis[i], self.vel[i]

def gather_pairs(root):
    pairs=[]
    for fam in ["FlatVel_A","Style_A"]:
        for sp in glob.glob(os.path.join(root,fam,"data","*.npy")):
            mp=sp.replace("data","model")
            if os.path.exists(mp): pairs.append((sp, mp))
    for fam in ["CurveFault_A","FlatFault_A"]:
        for sp in glob.glob(os.path.join(root,fam,"seis*_*.npy")):
            mp=os.path.join(root,fam,os.path.basename(sp).replace("seis","vel"))
            if os.path.exists(mp): pairs.append((sp, mp))
    return pairs

# ------------------------------------- net --------------------------------------- #
NORM={'bn': nn.BatchNorm2d}
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
        return self.out(x)

# --------------------------------- utility --------------------------------------- #
def denorm(t): return t*1500.0 + 1500.0

# --------------------------------- main ------------------------------------------ #
def main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    pairs=gather_pairs(args.data_root)
    ds   =SmallOpenFWI(pairs)
    rnd  = [0, 500, 1000, 1500, 999]
    seis = torch.stack([ds[i][0] for i in rnd]).to(device)
    vel_true = torch.stack([ds[i][1] for i in rnd]).to(device)

    # generator
    G = InversionNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    if isinstance(ckpt,dict) and "G" in ckpt:
        G.load_state_dict(ckpt["G"],strict=True)
    else:
        G.load_state_dict(ckpt,strict=True)
    G.eval()

    with torch.no_grad():
        vel_pred = G(seis)

    vel_pred = denorm(vel_pred.cpu())
    vel_true = denorm(vel_true.cpu())

    for i in range(args.num):
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].imshow(vel_true[i,0],cmap="jet"); ax[0].set_title("Ground truth (m/s)")
        ax[1].imshow(vel_pred[i,0],cmap="jet"); ax[1].set_title("Velocity-GAN prediction (m/s)")
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.show()

# ------------------------------------------------------------------------------- #
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True,help="Path to velocitygan .pt file")
    ap.add_argument("--data_root",default="./data",help="dataset root")
    ap.add_argument("--num",type=int,default=5,help="number of random samples")
    args=ap.parse_args()
    main(args)
