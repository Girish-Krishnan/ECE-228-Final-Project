#!/usr/bin/env python3
#  Fourier-DeepONet: Kaggle submission generator
from __future__ import annotations
import argparse, glob, math, os, csv
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

V_MIN, V_MAX = 1500.0, 4500.0
def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] → m s⁻¹"""
    return (x + 1.) * 0.5 * (V_MAX - V_MIN) + V_MIN

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--test_dir", required=True, help="directory with .npy seismic cubes")
    p.add_argument("--ckpt",     required=True, help="path to fourier_deeponet_best.pth")
    p.add_argument("--outfile",  default="submission.csv")
    p.add_argument("--width",    type=int, default=64,  help="model width used in training")
    p.add_argument("--modes",    type=int, default=20,  help="Fourier modes used in training")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

from fourier_deeponet_core import FourierDeepONet           # your definition

def main() -> None:
    args = cli()
    device = torch.device(args.device)

    net = FourierDeepONet(num_parameter=1,
                          width=args.width,
                          modes1=args.modes,
                          modes2=args.modes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["ema"] if "ema" in ckpt else ckpt["model_state_dict"]
    net.load_state_dict(state, strict=False)
    net.eval()

    header = ["oid_ypos"] + [f"x_{i}" for i in range(1, 70, 2)]

    test_files = sorted(glob.glob(os.path.join(args.test_dir, "*.npy")))
    if not test_files:
        raise SystemExit("✗  No .npy files found in --test_dir")

    with torch.no_grad(), open(args.outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for fp in tqdm(test_files, desc="Inferencing"):
            arr = np.load(fp).astype(np.float32)          # (5,1000,70)
            if arr.ndim != 3:
                raise ValueError(f"Unexpected shape in {fp}: {arr.shape}")

            arr = np.log1p(np.abs(arr)) * np.sign(arr) / math.log(61)
            branch = torch.from_numpy(arr).unsqueeze(0)   # (1,5,1000,70)
            trunk  = torch.zeros(1, 1)                    # dummy scalar

            pred_norm = net((branch.to(device), trunk.to(device)))   # (1,1,70,70)
            vel = denorm(pred_norm[0, 0]).cpu().numpy()              # (70,70) in m/s

            stem = Path(fp).stem                                # e.g. 000039dca2
            for y in range(70):
                vals = vel[y, 0:70:2]                           # odd columns 0,2,…68
                writer.writerow([f"{stem}_y_{y}"] +
                                 [f"{v:.1f}" for v in vals])

    print(f"✓ Submission saved to {args.outfile}")

if __name__ == "__main__":
    main()