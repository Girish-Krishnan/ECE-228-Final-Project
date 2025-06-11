import argparse, glob, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from math import ceil
from torchmetrics.functional import structural_similarity_index_measure as ssim

class SmallOpenFWI(torch.utils.data.Dataset):
    def __init__(self, pairs):
        s_lst, v_lst = [], []
        for sp, vp in pairs:
            s = np.load(sp).astype(np.float32)
            v = np.load(vp).astype(np.float32)
            if v.ndim == 3: v = v[:, None, :, :]
            s_lst.append(torch.from_numpy(s))
            v_lst.append(torch.from_numpy(v))
        self.seis = torch.cat(s_lst, 0)
        self.vel  = torch.clamp((torch.cat(v_lst, 0) - 1500.0) / 1500.0, -1, 1)
    def __len__(self):  return len(self.seis)
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

NORM = {'bn': nn.BatchNorm2d}

class ConvBlock(nn.Module):
    def __init__(self,i,o,k=3,s=1,p=1,n='bn'):
        super().__init__()
        layers=[nn.Conv2d(i,o,k,s,p)]
        if n in NORM: layers.append(NORM[n](o))
        layers.append(nn.LeakyReLU(0.2,True))
        self.layers=nn.Sequential(*layers)
    def forward(self,x): return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self,i,o,k=3,s=1,p=1,n='bn'):
        super().__init__()
        layers=[nn.Conv2d(i,o,k,s,p)]
        if n in NORM: layers.append(NORM[n](o))
        layers.append(nn.Tanh())
        self.layers=nn.Sequential(*layers)
    def forward(self,x): return self.layers(x)

class DeconvBlock(nn.Module):
    def __init__(self,i,o,k=2,s=2,p=0,n='bn'):
        super().__init__()
        layers=[nn.ConvTranspose2d(i,o,k,s,p)]
        if n in NORM: layers.append(NORM[n](o))
        layers.append(nn.LeakyReLU(0.2,True))
        self.layers=nn.Sequential(*layers)
    def forward(self,x): return self.layers(x)

class InversionNet(nn.Module):
    def __init__(self,d1=32,d2=64,d3=128,d4=256,d5=512,ratio=1.0):
        super().__init__()
        self.c1 = ConvBlock(5,d1,k=(7,1),s=(2,1),p=(3,0))
        self.c2_1=ConvBlock(d1,d2,k=(3,1),s=(2,1),p=(1,0)); self.c2_2=ConvBlock(d2,d2,k=(3,1),p=(1,0))
        self.c3_1=ConvBlock(d2,d2,k=(3,1),s=(2,1),p=(1,0)); self.c3_2=ConvBlock(d2,d2,k=(3,1),p=(1,0))
        self.c4_1=ConvBlock(d2,d3,k=(3,1),s=(2,1),p=(1,0)); self.c4_2=ConvBlock(d3,d3,k=(3,1),p=(1,0))
        self.c5_1=ConvBlock(d3,d3,s=2); self.c5_2=ConvBlock(d3,d3)
        self.c6_1=ConvBlock(d3,d4,s=2); self.c6_2=ConvBlock(d4,d4)
        self.c7_1=ConvBlock(d4,d4,s=2); self.c7_2=ConvBlock(d4,d4)
        self.c8   = ConvBlock(d4,d5,k=(8,ceil(70*ratio/8)),p=0)
        self.d1_1=DeconvBlock(d5,d5,k=5);           self.d1_2=ConvBlock(d5,d5)
        self.d2_1=DeconvBlock(d5,d4,k=4,p=1);       self.d2_2=ConvBlock(d4,d4)
        self.d3_1=DeconvBlock(d4,d3,k=4,p=1);       self.d3_2=ConvBlock(d3,d3)
        self.d4_1=DeconvBlock(d3,d2,k=4,p=1);       self.d4_2=ConvBlock(d2,d2)
        self.d5_1=DeconvBlock(d2,d1,k=4,p=1);       self.d5_2=ConvBlock(d1,d1)
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

def main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=SmallOpenFWI(gather_pairs(args.data_dir))
    model=InversionNet().to(device)
    state=torch.load(args.ckpt,map_location=device)
    if isinstance(state,dict) and "model_state_dict" in state:
        state=state["model_state_dict"]
    model.load_state_dict(state,strict=True)
    model.eval()

    idx = torch.linspace(0, len(ds)-1, steps=args.n).long()
    s_batch = torch.stack([ds[i][0] for i in idx]).to(device)
    v_true  = torch.stack([ds[i][1] for i in idx]).to(device)

    with torch.no_grad():
        pred = model(s_batch)

    v_pred = denorm(pred.cpu())
    v_true = denorm(v_true.cpu())

    mae = F.l1_loss(v_pred, v_true).item()
    rmse = torch.sqrt(F.mse_loss(v_pred, v_true)).item()
    ssim_val = ssim(v_pred, v_true).item()
    rel_l2 = (torch.norm(v_pred - v_true) / torch.norm(v_true)).item()

    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | SSIM: {ssim_val:.4f} | RelL2: {rel_l2:.4f}")

    for i in range(args.n):
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].imshow(v_true[i,0],cmap="jet",vmin=1500,vmax=3000); ax[0].set_title("Ground Truth")
        ax[1].imshow(v_pred[i,0],cmap="jet",vmin=1500,vmax=3000); ax[1].set_title("Prediction")
        for a in ax: a.axis("off")
        plt.tight_layout(); plt.show()

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True)
    ap.add_argument("--data_dir",default="./data")
    ap.add_argument("--n",type=int,default=5)
    args=ap.parse_args()
    main(args)
