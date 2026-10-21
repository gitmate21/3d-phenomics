"""Pheno4D dataset + collate for spconv (batch index is column 0 of the coords)."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from voxelize import voxelize_sample

SPLIT_FILES = {
    "train": "Pheno4D_train.txt",
    "val": "Pheno4D_val.txt",
    "trainval": "Pheno4D_trainval.txt",
    "test": "Pheno4D_test.txt",
}


def read_split(root: str, phase: str):
    path = os.path.join(root, "splits", SPLIT_FILES[phase])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/prepare_pheno4d.py first (it writes splits/).")
    with open(path) as fh:
        items = [ln.strip() for ln in fh if ln.strip()]
    if not items:
        raise ValueError(f"{path} is empty")
    return items


class Pheno4DDataset(Dataset):
    """Each item is one whole plant, voxelized (and augmented when ``train``).

    ``epoch_size`` makes a short split behave like a longer epoch by repeating it
    (random augmentation makes the repeats different).
    """

    def __init__(self, root: str, phase: str, conf, train: bool, seed: int = 0, repeat: int = 1):
        self.root, self.conf, self.train = root, conf, train
        self.items = read_split(root, phase)
        self.repeat = repeat if train else 1
        self.seed = seed

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, idx):
        rel = self.items[idx % len(self.items)]
        with np.load(os.path.join(self.root, rel)) as z:
            xyz, label = z["xyz"], z["label"]
        # Fresh entropy per call in training; eval voxelization is deterministic.
        rng = np.random.default_rng(None if self.train else self.seed + idx)
        s = voxelize_sample(xyz, label, self.conf, self.train, rng, name=rel)
        s["name"] = rel
        s.setdefault("xyz", None)      # eval: the points actually voxelized (soil already removed if dropped)
        return s


def collate(samples):
    """Concatenate voxels of a batch; coords become (M, 4) int32 = [batch, x, y, z]."""
    coords, feats, labels = [], [], []
    for b, s in enumerate(samples):
        c = torch.from_numpy(s["coords"]).int()
        coords.append(torch.cat([torch.full((len(c), 1), b, dtype=torch.int32), c], 1))
        feats.append(torch.from_numpy(s["feats"]))
        labels.append(torch.from_numpy(s["labels"]))
    coords = torch.cat(coords)
    return {
        "coords": coords,
        "feats": torch.cat(feats).float(),
        "labels": torch.cat(labels).long(),
        "batch_size": len(samples),
        "spatial_shape": (coords[:, 1:].max(0).values + 1).tolist(),
        # evaluation-only extras (lists, one entry per sample)
        "inverse": [s.get("inverse") for s in samples],
        "point_labels": [s.get("point_labels") for s in samples],
        "xyz": [s.get("xyz") for s in samples],
        "name": [s["name"] for s in samples],
    }
