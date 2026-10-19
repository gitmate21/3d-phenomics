# Eff-3DPSeg — spconv port

Port of [Eff-3DPSeg](https://arxiv.org/abs/2212.10263) (Luo et al., *Plant Phenomics*), a sparse-convolution
Res16UNet34C for stem/leaf segmentation of Pheno4D plant point clouds. The original used MinkowskiEngine on
torch 1.8 / CUDA 11.1 and was missing its config, dataset registry and splits; this port runs on current
PyTorch and Kaggle using [spconv](https://github.com/traveller59/spconv) (prebuilt wheels, no source build).

## What changed vs. the original
| Original | Port |
|---|---|
| MinkowskiEngine (source build, CUDA ≤ 11.1) | spconv v2, or a pure-PyTorch fallback (`scripts/models.py`, `scripts/sparse_torch.py`) |
| `ME.utils.sparse_quantize`, `lib/voxelizer.py`, `lib/transforms.py` | numpy/scipy (`scripts/voxelize.py`) |
| missing `config.py`, `splits/`, dataset registry | `config.py`, `prepare_pheno4d.py`, `scripts/datasets.py` |
| broken preprocessing script | `scripts/prepare_pheno4d.py` (plant-level splits) |
| DDP + wandb + fairseq utils | single GPU, AMP, plain log file |
| no resume of optimizer/iteration | full resume (`--resume auto`) + time limit for Kaggle |
| no inference | `scripts/infer.py` (full-resolution mIoU, `.ply` output) |
| pointnet2 CUDA ext, CRF, ScanNet/soybean code | dropped (unused by training) |

Architecture, layer/plane config and loss (cross-entropy, soil ignored) are unchanged.

## Not carried over
- **Pretrained weights**: the original fine-tunes from a pretrain checkpoint that is not distributed and is a
  MinkowskiEngine state dict, so it cannot be loaded here. Train from scratch, or pretrain yourself and pass `--weights`.
- The annotation-efficient *pretraining* stage is not in the original repo and is not implemented here.
- Mean-field CRF post-processing.

## Sparse backend
`scripts/models.py` uses spconv when it imports, otherwise `scripts/sparse_torch.py` (pure PyTorch: same
layers, works on any torch/CUDA and on CPU, but slower and more memory-hungry). Select with
`EFF_SPARSE_BACKEND=auto|spconv|torch` (default `auto`). spconv ships wheels only up to CUDA 12.6 and compiles
C++ at import time, which fails on CUDA 13 (current Colab), so the Colab/Kaggle notebooks use `torch`.
`python scripts/test_sparse_torch.py` checks the fallback layers against PyTorch's dense convolutions.

## Usage
```bash
export EFF_SPARSE_BACKEND=torch   # or install spconv-cuXXX (optional, faster) and leave it as auto
python scripts/smoke_test.py
python scripts/prepare_pheno4d.py --raw /path/to/Pheno4D --out data --val-plants Tomato06 --test-plants Tomato07
python scripts/train.py --data data --run-name demo --max-iter 2000
python scripts/train.py --data data --run-name demo --max-iter 2000 --resume auto
python scripts/infer.py --data data --ckpt checkpoints/demo_best.pth --phase test --out-dir preds --save-ply
```
On Kaggle use `notebooks/kaggle_train.ipynb`. Env overrides: `EFF_DATA_DIR`, `EFF_CKPT_DIR`, `EFF_LOG_DIR`.

## Things to verify with your data
- **Units / voxel size**: `--voxel-size` is in raw point-cloud units (original: 0.2). Check it gives a sensible
  voxel grid (plants of roughly 50–200 voxels tall).
- **Label column**: `prepare_pheno4d.py --label-col` (default 3); raw ids 0 → soil, 1 → stem, ≥2 → leaf.
- `conv1_kernel_size=5` and `bn_momentum=0.02` are the defaults of the upstream framework; the original `config.py` was not in the repo.
- Input features are constant (geometry only), as in the original (`feats = ones`); `--feature-mode xyz` is an option.

Original code is for research use; see the upstream repository for its license.
