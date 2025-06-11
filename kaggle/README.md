# Training Fourier DeepONet on the Entire OpenFWI Dataset for Kaggle Submission

This directory contains code for training a Fourier DeepONet model on the full OpenFWI dataset (\~622GB) using distributed training via `torchrun`, and generates a Kaggle-compatible `submission.csv` file for the [Waveform Inversion Kaggle competition](https://www.kaggle.com/competitions/waveform-inversion).

---

## Directory Overview

```
.
├── fourier_deeponet_core.py          # Model definition and dataset loader
├── distributed_training.py           # Main training script for distributed environment
└── kaggle_submission.py              # Inference + submission CSV generator
```

---

## Step 1: Training on Full OpenFWI Dataset (Distributed)

Firstly, you need a directory containing the full OpenFWI dataset. This code is designed to work with the entire dataset, which is quite large (over 622GB). 

Let's say your dataset is stored in `/mnt/` (on a large GPU cluster), in folders like:

```
/mnt/
  ├── FlatVel_A/data/*.npy
  ├── FlatVel_A/model/*.npy
  ├── CurveFault_A/seis_*.npy
  ├── CurveFault_A/vel_*.npy
  └── ... more folders for other variations
  ...
```

Use `torchrun` to launch distributed training. Example command:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    distributed_training.py \
    --root /mnt --fp16 --dist --batch 4 --accum 2 \
    --log /mnt/train.log --epochs 100 --lr 3e-3
```

### Key Arguments

* `--fp16`: Enable mixed precision training (saves memory and speeds up training)
* `--dist`: Enable distributed mode (required for multi-GPU)
* `--batch`: Per-GPU batch size
* `--accum`: Gradient accumulation steps (total batch size = batch $\times$ accum $\times$ nproc)
* `--log`: Path to CSV file for logging training/validation MAE
* `--root`: Root directory containing all the OpenFWI data folders

### Output Files

* `fourier_deeponet_best.pth`: Saved EMA model weights in `/mnt/`
* `train.log`: CSV training log with epoch-wise MAE (normalized and in m/s)

---

## Step 2: Generate Submission for Kaggle

Firstly, you must have the trained model checkpoint (`fourier_deeponet_best.pth`) from the previous step. Secondly, you need the test data from the Kaggle competition, which can be downloaded from the Kaggle competition page.

After training, use `kaggle_submission.py` to generate the submission file. Example:

```bash
python kaggle_submission.py \
    --test_dir /mnt/test_data \
    --ckpt /mnt/fourier_deeponet_best.pth \
    --outfile submission.csv \
    --width 64 \
    --modes 20
```

### Notes

* `--test_dir` must contain `.npy` seismic cubes (shape: 5x1000x70)
* Output `submission.csv` follows Kaggle's required format. See the competition page for details on how this is formatted.

---

## Step 3: Submit to Kaggle

Make sure your Kaggle API token is saved at `~/.kaggle/kaggle.json`. Then run:

```bash
kaggle competitions submit -c waveform-inversion -f submission.csv -m "Message"
```

where "Message" is a brief description of your submission (e.g., "Fourier DeepONet trained on full OpenFWI dataset").

---

## Code Summary

### `fourier_deeponet_core.py`

* Contains the full model architecture: `SpectralConv2d`, `UBlock`, `Decoder`, `BranchNet`, `TrunkNet`, `FourierDeepONet`
* Includes single-GPU `main()` training loop for smaller-scale experimentation

### `distributed_training.py`

* Loads and normalizes full OpenFWI dataset
* Uses `torchrun` for distributed training
* Applies mixed precision and EMA (Exponential Moving Average) for better generalization
* Logs metrics to CSV and saves best model checkpoint

### `kaggle_submission.py`

* Loads test seismic cubes, applies log-scaling
* Runs inference on each cube, extracts odd columns
* Saves results in Kaggle-compatible CSV format
