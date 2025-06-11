import argparse, glob, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_msssim import ssim
from tqdm import tqdm
from math import ceil
import matplotlib.pyplot as plt

class SmallOpenFWI(Dataset):
    def __init__(self, pairs):
        s, v = [], []
        for sp, vp in pairs:
            seis = np.load(sp).astype(np.float32)
            vel  = np.load(vp).astype(np.float32)
            if vel.ndim == 3: vel = vel[:, None]
            s.append(torch.from_numpy(seis))
            v.append(torch.clamp((torch.from_numpy(vel) - 1500.0) / 1500.0, -1, 1))
        self.s, self.v = torch.cat(s, 0), torch.cat(v, 0)

    def __len__(self): return len(self.s)
    def __getitem__(self, i): return self.s[i], self.v[i]

def gather_pairs(root):
    pairs=[]
    for fam in ["FlatVel_A","Style_A"]:
        for sp in glob.glob(os.path.join(root,fam,"data","*.npy")):
            mp = sp.replace("data","model")
            if os.path.exists(mp): pairs.append((sp, mp))
    for fam in ["CurveFault_A","FlatFault_A"]:
        for sp in glob.glob(os.path.join(root,fam,"seis*_*.npy")):
            mp = os.path.join(root,fam,os.path.basename(sp).replace("seis","vel"))
            if os.path.exists(mp): pairs.append((sp, mp))
    return pairs

NORM = { 'bn': nn.BatchNorm2d }

class ConvB(nn.Module):
    def __init__(self,i,o,k=3,s=1,p=1,n='bn'):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(i,o,k,s,p),
            *( [NORM[n](o)] if n in NORM else [] ),
            nn.LeakyReLU(0.2,True)
        )
    def forward(self,x): return self.seq(x)

class ConvB_Tanh(nn.Module):
    def __init__(self,i,o,k=3,s=1,p=1,n='bn'):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(i,o,k,s,p),
            *( [NORM[n](o)] if n in NORM else [] ),
            nn.Tanh()
        )
    def forward(self,x): return self.seq(x)

class DeconvB(nn.Module):
    def __init__(self,i,o,k=2,s=2,p=0,n='bn'):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(i,o,k,s,p),
            *( [NORM[n](o)] if n in NORM else [] ),
            nn.LeakyReLU(0.2,True)
        )
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
        self.d1_1=DeconvB(d5,d5,k=5);         self.d1_2=ConvB(d5,d5)
        self.d2_1=DeconvB(d5,d4,k=4,p=1);     self.d2_2=ConvB(d4,d4)
        self.d3_1=DeconvB(d4,d3,k=4,p=1);     self.d3_2=ConvB(d3,d3)
        self.d4_1=DeconvB(d3,d2,k=4,p=1);     self.d4_2=ConvB(d2,d2)
        self.d5_1=DeconvB(d2,d1,k=4,p=1);     self.d5_2=ConvB(d1,d1)
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
        return self.out(F.pad(x,[-5,-5,-5,-5]))

class Discriminator(nn.Module):
    def __init__(self,d1=32,d2=64,d3=128,d4=256):
        super().__init__()
        self.seq = nn.Sequential(
            ConvB(1,d1,s=2), ConvB(d1,d1),
            ConvB(d1,d2,s=2), ConvB(d2,d2),
            ConvB(d2,d3,s=2), ConvB(d3,d3),
            ConvB(d3,d4,s=2), ConvB(d4,d4),
            ConvB(d4,1,k=5,p=0)
        )
    def forward(self,x): return self.seq(x).view(x.size(0),-1)

def denorm(t): return t * 1500.0 + 1500.0

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs  = gather_pairs(args.data_root)
    ds     = SmallOpenFWI(pairs)
    n = len(ds)
    tr, va, te = int(0.7*n), int(0.2*n), n - int(0.7*n) - int(0.2*n)
    tr_ds, va_ds, te_ds = random_split(ds, [tr, va, te])
    tr_loader = DataLoader(tr_ds,batch_size=args.batch,shuffle=True,num_workers=2,pin_memory=True)
    va_loader = DataLoader(va_ds,batch_size=args.batch,shuffle=False,num_workers=2,pin_memory=True)

    G = InversionNet().to(device)
    D = Discriminator().to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))

    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    lam = 100.0

    for epoch in range(1, args.epochs+1):
        G.train(); D.train()
        g_mae = 0.0
        loop = tqdm(tr_loader,desc=f"Epoch {epoch}/{args.epochs}")
        for s, v in loop:
            s, v = s.to(device), v.to(device)
            real_lbl = torch.ones((s.size(0),1),device=device)
            fake_lbl = torch.zeros_like(real_lbl)

            with torch.no_grad(): fake = G(s)
            d_real, d_fake = D(v), D(fake.detach())
            loss_D = 0.5 * (bce(d_real,real_lbl) + bce(d_fake,fake_lbl))
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            fake = G(s)
            adv_loss = bce(D(fake), real_lbl)
            l1_loss  = l1(fake, v)
            loss_G   = adv_loss + lam * l1_loss
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            g_mae += l1_loss.item() * s.size(0)
            loop.set_postfix(D=loss_D.item(), G_L1=l1_loss.item())

        G.eval(); val_mae = 0.0
        with torch.no_grad():
            for s, v in va_loader:
                s, v = s.to(device), v.to(device)
                pred = G(s)
                val_mae += l1(pred, v).item() * s.size(0)
        val_mae /= len(va_loader.dataset)
        print(f"Epoch {epoch:2d} | Train MAE: {g_mae/len(tr_loader.dataset):.4f} | Val MAE: {val_mae:.4f}")

        if epoch % 5 == 0:
            torch.save({'G':G.state_dict(),'D':D.state_dict()}, f"velocitygan_epoch{epoch}.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    main(args)
