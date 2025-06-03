import argparse, glob, os
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# Dataset helpers  (unchanged – same transforms used in training)
# ──────────────────────────────────────────────────────────────
class SeismicDataset(torch.utils.data.Dataset):
    """Returns (branch, trunk_dummy, target_norm)"""
    def __init__(self, pair_list):
        branches, targets = [], []
        for s_path, v_path in pair_list:
            s = np.load(s_path).astype(np.float32)                              # (N,5,1000,70)
            v = np.load(v_path).astype(np.float32)                              # (N,1,70,70) or (N,70,70)
            if v.ndim == 3: v = v[:, None]
            branches.append(torch.from_numpy(s))
            targets.append(torch.from_numpy(v))
        self.branch = torch.cat(branches)
        self.target = torch.cat(targets)
        self.trunk  = torch.zeros(len(self.branch), 1)                          # dummy
        self.target = torch.clamp((self.target - 1500) / 3000, -1, 1)           # match training

    def __len__(self):  return len(self.branch)
    def __getitem__(self, i):
        return self.branch[i], self.trunk[i], self.target[i]

def gather_pairs(root):
    pairs=[]
    for fam in ["FlatVel_A","Style_A"]:
        for sp in glob.glob(os.path.join(root,fam,"data","*.npy")):
            mp = sp.replace("data","model");          pairs.append((sp,mp))
    for fam in ["CurveFault_A","FlatFault_A"]:
        for sp in glob.glob(os.path.join(root,fam,"seis*_*.npy")):
            mp=os.path.join(root,fam,os.path.basename(sp).replace("seis","vel"))
            pairs.append((sp,mp))
    return pairs

# ──────────────────────────────────────────────────────────────
# Fourier-DeepONet  (EXACT layer names! – do NOT edit)
# ──────────────────────────────────────────────────────────────
class SpectralConv2d(nn.Module):
    def __init__(self, inc,outc,m1,m2):
        super().__init__(); self.modes1,self.modes2=m1,m2
        scale=1/(inc*outc)
        self.weights1=nn.Parameter(scale*torch.randn(inc,outc,m1,m2,dtype=torch.cfloat))
        self.weights2=nn.Parameter(scale*torch.randn(inc,outc,m1,m2,dtype=torch.cfloat))
    @staticmethod
    def c_mul(a,b): return torch.einsum("bixy,ioxy->boxy",a,b)
    def forward(self,x):
        B=x.size(0); m1,m2=self.modes1,self.modes2
        x_ft=torch.fft.rfftn(x,dim=[-2,-1])
        out_ft=torch.zeros(B,self.weights1.shape[1],x.size(-2),x.size(-1)//2+1,
                           dtype=torch.cfloat,device=x.device)
        out_ft[:,:, :m1,:m2]=self.c_mul(x_ft[:,:, :m1,:m2], self.weights1)
        out_ft[:,:,-m1:,:m2]=self.c_mul(x_ft[:,:,-m1:,:m2], self.weights2)
        return torch.fft.irfftn(out_ft,s=x.shape[-2:])

class UBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.down1=nn.Conv2d(c,c,3,2,1); self.down2=nn.Conv2d(c,c,3,2,1)
        self.up1  =nn.ConvTranspose2d(c,c,4,2,1)
        self.up2  =nn.ConvTranspose2d(c*2,c,4,2,1)
    def forward(self,x):
        d1=F.relu(self.down1(x)); d2=F.relu(self.down2(d1))
        u1=F.relu(self.up1(d2));  u2=F.relu(self.up2(torch.cat([u1,d1],1)))
        return u2

class Decoder(nn.Module):
    def __init__(self,m1,m2,w):
        super().__init__()
        self.conv0=SpectralConv2d(w,w,m1,m2); self.conv1=SpectralConv2d(w,w,m1,m2)
        self.w0=nn.Conv1d(w,w,1); self.w1=nn.Conv1d(w,w,1)
        self.unet=UBlock(w); self.fc1=nn.Linear(w,128); self.fc2=nn.Linear(128,1)
    def forward(self,x):
        B,C,H,W=x.shape
        x=F.relu(self.conv0(x)+self.w0(x.view(B,C,-1)).view(B,C,H,W))
        x=F.relu(self.conv1(x)+self.w1(x.view(B,C,-1)).view(B,C,H,W)+self.unet(x))
        x=F.adaptive_avg_pool2d(x,(70,70)); x=x.permute(0,2,3,1)
        x=F.relu(self.fc1(x)); return self.fc2(x).permute(0,3,1,2)

class BranchNet(nn.Module):
    def __init__(self,w): super().__init__(); self.fc0=nn.Linear(5,w)
    def forward(self,x):
        x=F.pad(x,(1,1,0,0)); x=x.permute(0,3,2,1); x=self.fc0(x); return x.permute(0,3,1,2)

class TrunkNet(nn.Module):
    def __init__(self,w,np): super().__init__(); self.fc0=nn.Linear(np,w)
    def forward(self,x): return self.fc0(x)[:,:,None,None]

class FourierDeepONet(nn.Module):
    def __init__(self,nparam=1,width=64,m1=20,m2=20):
        super().__init__()
        self.branch=BranchNet(width); self.trunk=TrunkNet(width,nparam)
        self.decoder=Decoder(m1,m2,width); self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,inp):
        xb=self.branch(inp[0]); xt=self.trunk(inp[1])
        return self.decoder(xb*xt + self.bias)

# ──────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────
def denorm(x): return x*3000+1500                                   # back to m/s

# --------------------------------------------------------------
def main(args):
    os.makedirs(args.outdir,exist_ok=True)

    # ---------------- dataset slice
    pairs   = gather_pairs(args.data_dir)
    ds      = SeismicDataset(pairs)
    idxs    = [0, 500, 1000, 1500, 999][:args.num]                  # clamp if fewer
    batch   = [ds[i] for i in idxs]
    branch  = torch.stack([b[0] for b in batch]).to(DEVICE)
    trunk   = torch.stack([b[1] for b in batch]).to(DEVICE)
    truth_n = torch.stack([b[2] for b in batch]).to(DEVICE)

    # ---------------- model
    net = FourierDeepONet().to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    net.load_state_dict(ckpt["model_state_dict"], strict=True)
    net.eval()

    with torch.no_grad():
        pred_n = net((branch,trunk))

    pred   = denorm(pred_n.cpu())
    truth  = denorm(truth_n.cpu())
    err    = torch.abs(pred - truth)                                # (B,1,70,70)

    # ----------------------------------------------------------
    # 1⃣  Triptych + error map
    # ----------------------------------------------------------
    for k in range(len(idxs)):
        fig,axs=plt.subplots(1,4,figsize=(18,4))
        # Stack all 5 input channels horizontally (shape: [1000, 5×70])
        stacked_gather = torch.cat([branch[k, i] for i in range(5)], dim=1)  # (1000, 350)
        axs[0].imshow(stacked_gather.cpu(), cmap="gray", aspect="auto")
        axs[0].set_title("Input gather (stacked channels)")

        axs[1].imshow(truth[k,0],cmap="jet");          axs[1].set_title("Truth (m/s)")
        axs[2].imshow(pred[k,0],cmap="jet");           axs[2].set_title("Prediction (m/s)")
        im=axs[3].imshow(err[k,0],cmap="inferno");     axs[3].set_title("|Error| (m/s)")
        for a in axs: a.axis('off')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axs[3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, label="|Error| (m/s)")
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/sample{k:02d}_triptych.png",dpi=300)
        plt.close(fig)

    # ----------------------------------------------------------
    # 2⃣  Error-vs-velocity scatter (all pixels, all samples)
    # ----------------------------------------------------------
    vel_vals = truth.view(-1).numpy()
    err_vals = err.view(-1).numpy()
    fig = plt.figure(figsize=(5,4))
    plt.hexbin(vel_vals, err_vals, gridsize=60, cmap="magma", bins='log')
    plt.xlabel("True velocity (m/s)"); plt.ylabel("|Error| (m/s)")
    plt.title("Error vs velocity")
    cb=plt.colorbar(); cb.set_label("log10(count)")
    fig.tight_layout(); fig.savefig(f"{args.outdir}/scatter_err_vs_vel.png",dpi=300); plt.close(fig)

    # ----------------------------------------------------------
    # 3⃣  Depth profile  – mean |error| per depth
    # ----------------------------------------------------------
    depth_err = err.mean(dim=[0,1,3]).numpy()                       # (70,)
    fig = plt.figure(figsize=(4,4))
    plt.plot(depth_err, np.arange(70))
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |error| (m/s)"); plt.ylabel("Depth / row index")
    plt.title("Depth profile of error")
    fig.tight_layout(); fig.savefig(f"{args.outdir}/depth_profile.png",dpi=300); plt.close(fig)

    # ----------------------------------------------------------
    # 4⃣  Mean log-amplitude residual spectrum (w/ proper axes)
    # ----------------------------------------------------------
    resid = (truth - pred).numpy()
    spec  = np.fft.fft2(resid, axes=(-2, -1))
    amp   = np.log10(np.abs(spec) + 1e-6)
    amp_mean = amp.mean(0)[0]  # shape (70,70)

    # Create frequency axes
    fx = np.fft.fftfreq(amp_mean.shape[1]) * amp_mean.shape[1]
    fy = np.fft.fftfreq(amp_mean.shape[0]) * amp_mean.shape[0]
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))

    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(np.fft.fftshift(amp_mean), extent=[fx.min(), fx.max(), fy.max(), fy.min()],
                   cmap="viridis", aspect='auto')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("log10 amplitude")
    ax.set_xlabel("Spatial frequency (X)")
    ax.set_ylabel("Spatial frequency (Y)")
    ax.set_title("Mean log-amplitude residual spectrum")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/residual_spectrum.png", dpi=300)
    plt.close(fig)


    print(f"✓  All figures saved in →  {args.outdir}/")

# --------------------------------------------------------------
if __name__ == "__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--ckpt", required=True, help="checkpoint (.pt/.pth)")
    pa.add_argument("--data_dir", default="./data")
    pa.add_argument("--num", type=int, default=5, help="#samples to visualise (≤5)")
    pa.add_argument("--outdir", default="./viz_out", help="directory for png files")
    args=pa.parse_args()
    main(args)
