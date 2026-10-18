"""
Quick end-to-end sanity check (synthetic data, no Pheno4D needed).

    python scripts/smoke_test.py

Steps 1-3 need only numpy/scipy. Steps 4-6 need torch + spconv + a CUDA GPU and are
skipped (with a message) otherwise.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from config import conf  # noqa: E402
from voxelize import voxelize_sample, quantize  # noqa: E402


def synthetic_plant(n=6000, seed=0):
    """A vertical stem (label 1), leaves (label 2) and a soil disk (label 0)."""
    rng = np.random.default_rng(seed)
    stem = np.c_[rng.normal(0, 0.8, n // 3), rng.normal(0, 0.8, n // 3), rng.uniform(0, 40, n // 3)]
    leaf = np.c_[rng.uniform(-15, 15, n // 3), rng.uniform(-15, 15, n // 3), rng.uniform(15, 40, n // 3)]
    soil = np.c_[rng.uniform(-20, 20, n // 3), rng.uniform(-20, 20, n // 3), rng.normal(0, 0.3, n // 3)]
    xyz = np.vstack([stem, leaf, soil]).astype(np.float32)
    lab = np.r_[np.ones(n // 3), np.full(n // 3, 2), np.zeros(n // 3)].astype(np.uint8)
    return xyz, lab


def main():
    print(conf.summary())
    xyz, lab = synthetic_plant()

    print("\n[1/6] voxelization (eval, deterministic)")
    rng = np.random.default_rng(0)
    s = voxelize_sample(xyz, lab, conf, train=False, rng=rng)
    assert s["coords"].shape[1] == 3 and s["coords"].dtype == np.int32
    assert len(s["inverse"]) == len(xyz) and s["inverse"].max() == len(s["coords"]) - 1
    assert set(np.unique(s["labels"])) <= {0, 1, 255}
    print(f"  {len(xyz)} points -> {len(s['coords'])} voxels, labels {np.unique(s['labels'])}")

    print("[2/6] augmentation (train)")
    t = voxelize_sample(xyz, lab, conf, train=True, rng=rng)
    assert t["coords"].min() == 0 and len(t["coords"]) > 0
    print(f"  train voxels: {len(t['coords'])}")

    print("[3/6] prepare_pheno4d + dataset round-trip")
    import prepare_pheno4d
    with tempfile.TemporaryDirectory() as tmp:
        raw, out = os.path.join(tmp, "raw"), os.path.join(tmp, "out")
        for plant in ("TomatoA", "TomatoB"):
            os.makedirs(os.path.join(raw, plant))
            for day in ("01", "02"):
                x, l = synthetic_plant(seed=hash((plant, day)) % 1000)
                np.savetxt(os.path.join(raw, plant, f"{plant}_{day}_a.txt"), np.c_[x, l])
        assert prepare_pheno4d.main(["--raw", raw, "--out", out, "--val-plants", "TomatoB"]) == 0
        try:
            import torch
        except ImportError:
            print("  torch not installed -> stopping after step 3")
            print("OK (partial)")
            return
        from datasets import Pheno4DDataset, collate
        ds = Pheno4DDataset(out, "train", conf, train=True)
        batch = collate([ds[0], ds[1]])
        assert batch["coords"].shape[1] == 4 and batch["coords"].dtype == torch.int32
        print(f"  batch coords {tuple(batch['coords'].shape)} spatial_shape {batch['spatial_shape']}")

    try:
        import spconv.pytorch  # noqa: F401
    except ImportError:
        print("spconv not installed -> skipping model steps. OK (partial)")
        return
    if not torch.cuda.is_available():
        print("no CUDA GPU -> skipping model steps. OK (partial)")
        return

    from models import build_model, to_sparse
    device = torch.device("cuda")

    print("[4/6] model forward")
    model = build_model(conf.model, 1, conf.num_classes, conf.conv1_kernel_size, conf.bn_momentum).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.2f}M parameters")
    out = model(to_sparse(batch, device))
    assert out.features.shape == (len(batch["coords"]), conf.num_classes), out.features.shape
    assert torch.equal(out.indices.cpu(), batch["coords"]), "output voxels must equal input voxels"

    print("[5/6] loss + backward + optimizer step")
    labels = batch["labels"].to(device)
    loss = torch.nn.functional.cross_entropy(out.features.float(), labels, ignore_index=conf.ignore_label)
    assert torch.isfinite(loss), loss
    loss.backward()
    torch.optim.SGD(model.parameters(), lr=1e-3).step()
    print(f"  loss {loss.item():.4f}")

    print("[6/6] AMP forward")
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(to_sparse(batch, device))
    assert torch.isfinite(out.features).all()
    print("OK")


if __name__ == "__main__":
    main()
