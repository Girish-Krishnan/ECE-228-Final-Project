#!/usr/bin/env python3
#  Fourier-DeepONet  –  distributed trainer with CSV logging
from __future__ import annotations
import argparse, glob, os, math, random, time, csv
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, distributed
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data / model
    p.add_argument('--root',  default='.', help='dataset root')
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--modes', type=int, default=20)
    # optimization
    p.add_argument('--epochs',   type=int,   default=80)
    p.add_argument('--lr',       type=float, default=3e-4)
    p.add_argument('--batch',    type=int,   default=4, help='per-GPU batch')
    p.add_argument('--accum',    type=int,   default=1, help='grad-accum steps')
    p.add_argument('--patience', type=int,   default=15, help='early-stop')
    p.add_argument('--workers',  type=int,   default=4)
    p.add_argument('--fp16',     action='store_true')
    # logging
    p.add_argument('--log',      default='training_log.csv',
                   help='CSV file to append training curves')
    # distributed
    p.add_argument('--dist',       action='store_true', help='expect torchrun')
    p.add_argument('--local_rank', type=int,
                   default=os.getenv("LOCAL_RANK", -1))
    return p.parse_args()

def set_seed(seed:int=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

V_MIN, V_MAX = 1500.0, 4500.0
def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] → physical m/s   (x is on the current device)"""
    return (x + 1.) * 0.5 * (V_MAX - V_MIN) + V_MIN

def collect_pairs(root:str) -> list[tuple[str,str,int]]:
    pairs=[]
    multi_folders = ("FlatVel_A","FlatVel_B","CurveVel_A","CurveVel_B",
                     "Style_A","Style_B")
    for fam in multi_folders:
        d,m = Path(root,fam,"data"), Path(root,fam,"model")
        if not d.is_dir(): continue
        for dp in sorted(d.glob("*.npy")):
            mp = m/dp.name.replace("data","model")
            if not mp.exists(): continue
            n = np.load(dp,mmap_mode="r").shape[0]
            pairs.extend([(str(dp),str(mp),i) for i in range(n)])
    for fam in ("CurveFault_A","CurveFault_B","FlatFault_A","FlatFault_B"):
        r = Path(root,fam)
        for sp in r.glob("seis*_*.npy"):
            vp = r/sp.name.replace("seis","vel")
            if vp.exists(): pairs.append((str(sp),str(vp),0))
    if not pairs: raise SystemExit("✗  No data found – check --root")
    return pairs

class BigFWI(Dataset):
    def __init__(self, pairs): self.items=pairs
    def __len__(self): return len(self.items)

    @staticmethod
    @lru_cache(maxsize=256)
    def _open(path:str): return np.load(path,mmap_mode='r')

    def __getitem__(self, idx:int):
        sp,vp,row = self.items[idx]
        seis = self._open(sp)[row]             # (5,1000,70)
        vel  = self._open(vp)[row][None,...] if self._open(vp)[row].ndim==2 \
               else self._open(vp)[row]        # (1,70,70)
        seis = np.log1p(abs(seis))*np.sign(seis)/math.log(61)          # [-1,1]
        vel  = np.clip((vel-V_MIN)/(V_MAX-V_MIN)*2-1, -1, 1)
        return (torch.from_numpy(seis.astype(np.float32)),
                torch.zeros(1,dtype=torch.float32),                   # dummy trunk
                torch.from_numpy(vel.astype(np.float32)))


from fourier_deeponet_core import FourierDeepONet

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay=decay
        self.shadow={k:v.clone().detach() for k,v in model.state_dict().items()}
    def update(self, model):
        with torch.no_grad():
            for k,v in model.state_dict().items():
                self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
    def copy_to(self,model): model.load_state_dict(self.shadow,strict=False)

def main():
    args = cli(); set_seed(0)
    dist = args.dist or int(os.getenv("WORLD_SIZE","1"))>1
    if dist and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")

    local_rank = int(os.getenv("LOCAL_RANK", args.local_rank if args.local_rank>=0 else 0))
    global_rank= int(os.getenv("RANK",0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda",local_rank) if torch.cuda.is_available() else torch.device("cpu")

    full_ds   = BigFWI(collect_pairs(args.root))
    n_total   = len(full_ds)
    n_val     = n_test = int(0.1*n_total)
    n_train   = n_total - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(0)
    )

    train_s = distributed.DistributedSampler(train_ds, shuffle=True) if dist else None
    val_s   = distributed.DistributedSampler(val_ds,   shuffle=False) if dist else None
    test_s  = distributed.DistributedSampler(test_ds,  shuffle=False) if dist else None

    dl_k = dict(batch_size=args.batch,
                num_workers=args.workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=args.workers>0)

    train_dl = DataLoader(train_ds, sampler=train_s, shuffle=train_s is None, **dl_k)
    val_dl   = DataLoader(val_ds,   sampler=val_s,   shuffle=False, **dl_k)
    test_dl  = DataLoader(test_ds,  sampler=test_s,  shuffle=False, **dl_k)

    net = FourierDeepONet(num_parameter=1,width=args.width,
                          modes1=args.modes,modes2=args.modes).to(device)
    if dist: net = DDP(net, device_ids=[local_rank] if torch.cuda.is_available() else None,
                       broadcast_buffers=False)

    ema   = EMA(net.module if dist else net)
    opt   = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler= GradScaler(enabled=args.fp16)
    crit  = nn.L1Loss()

    ckpt_path = "/mnt/fourier_deeponet_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if 'ema' in ckpt:
            ema.shadow.update(ckpt['ema'])  # Load EMA weights
            (net.module if dist else net).load_state_dict(ckpt['ema'], strict=False)
            print("✓ Resumed training from best checkpoint.")

    if global_rank==0 and not Path(args.log).exists():
        with open(args.log,'w',newline='') as f:
            csv.writer(f).writerow(["epoch","train_mae","val_mae","val_mae_ms"])

    best, patience = float('inf'), 0

    for ep in range(1,args.epochs+1):
        if train_s: train_s.set_epoch(ep)
        net.train(); running=0.; opt.zero_grad(set_to_none=True)

        loop=tqdm(train_dl,disable=global_rank!=0,desc=f"Epoch {ep}/{args.epochs}")
        for step,(br,tr,tar) in enumerate(loop,1):
            br,tr,tar = br.to(device), tr.to(device), tar.to(device)
            with autocast(enabled=args.fp16):
                pred = net((br,tr))
                loss = crit(pred,tar)/args.accum
            scaler.scale(loss).backward()
            if step%args.accum==0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                ema.update(net.module if dist else net)
            running += loss.item()*br.size(0)*args.accum
            loop.set_postfix(mae=loss.item()*args.accum)

        train_mae = running/len(train_dl.dataset)

        net.eval(); ema.copy_to(net.module if dist else net)
        val_sum=0.; val_ms=0.
        if val_s: val_s.set_epoch(ep)
        with torch.no_grad():
            for br,tr,tar in val_dl:
                br,tr,tar = br.to(device),tr.to(device),tar.to(device)
                pred = net((br,tr))
                val_sum += crit(pred,tar).item()*br.size(0)
                val_ms  += torch.mean(torch.abs(denorm(pred)-denorm(tar))).item()*br.size(0)
        val_mae = val_sum/len(val_dl.dataset)
        val_ms  = val_ms /len(val_dl.dataset)

        if global_rank==0:
            print(f"Ep {ep:3d} • train {train_mae:.4f} • val {val_mae:.4f} • {val_ms:.1f} m/s")
            with open(args.log,'a',newline='') as f:
                csv.writer(f).writerow([ep, f"{train_mae:.6f}", f"{val_mae:.6f}",
                                        f"{val_ms:.3f}"])
            if val_mae<best:
                best,patience = val_mae,0
                torch.save({"ema":ema.shadow}, "/mnt/fourier_deeponet_best.pth")
                print("   ✓ new best saved")
            else:
                patience+=1
                if patience>=args.patience:
                    print("Early stopping."); break

        # restore raw weights for next epoch
        if ep!=args.epochs:
            (net.module if dist else net).load_state_dict(ema.shadow,strict=False)
        if dist: torch.distributed.barrier()

    net.eval(); ema.copy_to(net.module if dist else net)
    test_sum=0.; test_ms=0.
    with torch.no_grad():
        for br,tr,tar in test_dl:
            br,tr,tar = br.to(device),tr.to(device),tar.to(device)
            pred = net((br,tr))
            test_sum += crit(pred,tar).item()*br.size(0)
            test_ms  += torch.mean(torch.abs(denorm(pred)-denorm(tar))).item()*br.size(0)
    test_mae = test_sum/len(test_dl.dataset)
    test_ms  = test_ms /len(test_dl.dataset)

    if global_rank==0:
        print(f"TEST •  MAE {test_mae:.4f}  •  {test_ms:.1f} m/s  (unnorm)")
        with open(args.log,'a',newline='') as f:
            csv.writer(f).writerow(["test","-",f"{test_mae:.6f}",f"{test_ms:.3f}"])

    if dist and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__=="__main__":
    main()