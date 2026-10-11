#!/usr/bin/env python
"""
Smoke test: validate the model + data pipeline end-to-end on the demo `.h5`.

Run this FIRST on Kaggle/Colab to confirm everything works before a long training:

    python scripts/smoke_test.py

Checks:
  1. Imports resolve.
  2. VCNN builds and runs a forward pass on a random 32^3 volume.
  3. Output shapes are correct: classification [N, K], embedding [N, 32].
  4. disc_loss is finite on synthetic embeddings.
  5. PartDataset loads the demo file and yields a correctly-shaped batch.
  6. One optimizer step runs and reduces/keeps loss finite.
"""

import os
import sys

import numpy as np
import torch
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from config import conf
from datasets import PartDataset
from torch.utils.data import DataLoader
from vcnnmodel import VCNN, disc_loss

import torch.nn.functional as F


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        raise AssertionError(msg)


def main():
    print("===> 1. config")
    print(conf.summary())
    check(conf.num_classes == 2, "num_classes == 2")

    print("===> 2. model forward on random 32^3 volume")
    model = VCNN(conf.num_classes, conf.criterion).to(conf.device)
    x = torch.rand(1, 1, 32, 32, 32, device=conf.device)
    out_cls, out_fts = model(x)
    check(out_cls.shape[1] == conf.num_classes,
          f"classification output has {conf.num_classes} channels")
    check(out_fts.shape[1] == 32, "embedding output has 32 dims")
    print(f"    cls {tuple(out_cls.shape)}  fts {tuple(out_fts.shape)}")

    print("===> 3. disc_loss finite on synthetic data")
    fts = torch.randn(100, 32, device=conf.device, requires_grad=True)
    y = torch.randint(0, 4, (100,), device=conf.device)
    loss = disc_loss(fts, y, conf.disc_loss_conf)
    loss.backward()
    check(torch.isfinite(loss).item(), "disc_loss is finite and differentiable")

    print("===> 4. dataset loads demo file")
    data_path = os.path.join(conf.data_dir, "x_y1_y2_idx_33_withoutnoise.h5")
    check(os.path.exists(data_path), f"demo data present: {data_path}")
    ds = PartDataset(data_path, 2, 1, train=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    image, label, vox_xyz, scale, xyz_o = next(iter(loader))
    check(image.dim() == 5, "image is a 5-D volume (B,C,D,H,W)")
    check(label.shape[-1] == 3, "label has 3 columns [cls, part, roi]")
    print(f"    image {tuple(image.shape)}  label {tuple(label.shape)}")

    print("===> 5. one training step")
    model.train()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    image = image.to(conf.device).float()
    label = label.to(conf.device)
    opt.zero_grad()
    out_cls, out_fts = model(image)
    roi = label[0, ..., -1]
    out_cls = out_cls.contiguous().index_select(0, roi)
    out_fts = out_fts.contiguous().index_select(0, roi)
    cls_loss = F.nll_loss(out_cls, label[0, ..., 0])
    part_loss = disc_loss(out_fts, label[0, ..., 1], conf.disc_loss_conf)
    total = cls_loss + part_loss
    total.backward()
    opt.step()
    check(torch.isfinite(total).item(),
          f"one full step OK (loss={total.item():.4f})")

    print("\nALL SMOKE TESTS PASSED ✔")


if __name__ == "__main__":
    main()
