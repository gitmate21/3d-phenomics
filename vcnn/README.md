<!-- # VCNN — Voxel-based CNN for Maize Structural Component Separation

A clean, cloud-ready reproduction of the training pipeline from:

> Jin S, Su Y, Gao S, Wu F, Ma Q, Xu K, Ma Q, Hu T, Liu J, Pang S, Guan H, Zhang J, Guo Q. 2019.
> *Separating the structural components of maize for field phenotyping using terrestrial lidar data and deep convolutional neural networks.*
> IEEE Transactions on Geoscience and Remote Sensing.

The model is a **3D U-Net (V-Net)** with two heads that, from a voxelized terrestrial-LiDAR
scan of a single maize plant, performs:
1. **Classification** (semantic segmentation): plant vs. background, per point.
2. **Segmentation** (instance segmentation): separate each individual leaf/stem.

This repo is a **modernized port** of the authors' original PyTorch 0.1.12 code:
- Modern PyTorch (>=1.x) APIs, no `Variable`, no `.data[0]`, no `embed()` breakpoints.
- Runs unchanged on **Kaggle GPU**, **Colab GPU**, or **local**.
- Central `config.py` for all paths/hyper-parameters (no hard-coded absolute paths).
- Robust **checkpoint / resume** so long training survives Kaggle/Colab session limits.

> **Use GPU, not TPU.** The 3D convolutions, variable-sized input volumes, data-dependent
> instance loop in the loss, and `cKDTree`/`np.unique` preprocessing are not TPU/XLA-friendly.

---

## Repository layout

```
vcnn/
├── config.py                 # central config + platform-aware paths
├── requirements.txt          # extra deps (torch comes from the platform)
├── .gitignore                # keeps checkpoints/large .h5 out of git
├── data/
│   ├── data.md               # dataset format documentation
│   └── x_y1_y2_idx_33_withoutnoise.h5   # tiny demo dataset (33 plants)
├── model/
│   └── model.md              # notes on pretrained / training your own
├── scripts/
│   ├── vcnnmodel.py          # 3D U-Net + discriminative loss
│   ├── datasets.py           # PyTables dataset + voxelization
│   ├── h5.py                 # HDF5 helpers
│   ├── utils.py              # weight init, dice loss, plotting
│   ├── train.py              # training entry point
│   └── smoke_test.py         # end-to-end validation (run this first)
└── notebooks/
    └── kaggle_train.ipynb    # Kaggle: clone -> install -> smoke test -> train
```

---

## Quick start (Kaggle GPU — recommended)

1. **Settings → Accelerator → GPU** (T4 x2 or P100).
2. **Settings → Internet → On** — required for `pip install` (Kaggle disables internet
   by default; the first time you'll do a one-time phone verification).
   Without this, `pip install` fails with *"Could not find a version that satisfies
   the requirement ..."*.
3. Open `notebooks/kaggle_train.ipynb`, set the clone URL to **your** GitHub fork.
4. Run the cells top to bottom: clone → install extras → `smoke_test.py` → `train.py`.

From the command line (inside the notebook or a terminal):

```bash
# install only the missing extras (torch is preinstalled on Kaggle/Colab).
# If this fails, confirm Internet is enabled in Settings.
pip install -r requirements.txt

# 1) validate everything works (fast)
python scripts/smoke_test.py

# 2) train on the bundled demo data
python scripts/train.py --data data/x_y1_y2_idx_33_withoutnoise.h5 --model vcnn_demo --epochs 5
```

### Resume after a session cut
```bash
python scripts/train.py --data <data>.h5 --model vcnn_demo --epochs 20 --resume 5
```

---

## Using your own data

Your `.h5` must contain a group `/process/` with arrays `x (N,3)`, `y1 (N,)`,
`y2 (N,)`, `idxs (M,2)` — see [`data/data.md`](data/data.md).

- **Kaggle:** upload the `.h5` as a Kaggle Dataset, then pass its path:
  `python scripts/train.py --data /kaggle/input/<dataset>/<file>.h5 ...`
- **Colab:** mount Google Drive and pass the path, or set `VCNN_DATA_DIR`.

All output locations can be overridden with env vars:
`VCNN_DATA_DIR`, `VCNN_CKPT_DIR`, `VCNN_LOG_DIR`.

---

## Configuration

Edit [`config.py`](config.py) for defaults (learning rate, epochs, loss weights, etc.).
Key values follow the paper:
- `num_classes = 2`, `criterion = "nll"`, `learning_rate = 1e-5`, `batch_size = 1`.
- `disc_loss_conf = {c0: 1, c1: 3, abcd: [1, 1, 0.01, 0.01]}` — the discriminative
  instance-embedding loss weights.

---

## What this Phase 0 / Phase 1 port includes

- [x] Cloud-ready repo scaffolding (`.gitignore`, `requirements.txt`, `config.py`).
- [x] Modern-PyTorch port of the model, dataset, loss, and training loop.
- [x] Checkpoint save / `--resume` for session survival.
- [x] End-to-end `smoke_test.py` and a Kaggle notebook.

### Not yet included (later phases)
- [ ] Data **preprocessing**: raw TLS point cloud → voxelized `.h5` (x/y1/y2/idxs).
- [ ] **Inference / testing** script (predict per point, map back to raw points).
- [ ] **Post-processing**: cluster embeddings → stem/leaf instances.
- [ ] **Evaluation**: instance-level & point-level F-score.

---

## License / citation

The original code is for learning use only; commercial use is not allowed by the
authors. If you use this work, please cite the paper above. -->
