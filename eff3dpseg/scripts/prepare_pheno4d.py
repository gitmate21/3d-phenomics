"""
Convert raw Pheno4D annotated text files into .npz point clouds + split lists.

Replaces the broken lib/datasets/preprocessing/pheno4d.py of the original repo.

Expected raw layout (Pheno4D):
    <raw>/Tomato01/Tomato01_03_a.txt      # annotated clouds end in '_a.txt'
    <raw>/Tomato02/...
Each row: x y z <semantic label> [more columns]. The label column is configurable
(--label-col, default 3). Raw label ids are mapped to
    0 -> 0 (soil), 1 -> 1 (stem), >=2 -> 2 (leaf)
(Pheno4D numbers individual leaves 2, 3, 4, ...; they are all 'leaf' here.)

Output:
    <out>/<Plant>/<name>.npz            # xyz float32 (N,3), label uint8 (N,)
    <out>/splits/Pheno4D_{train,val,trainval,test}.txt   # paths relative to <out>

Splits are by plant (never by day) so the same plant never appears in two splits.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

POINTCLOUD_SUFFIX = "_a.txt"


def read_annotated_txt(path: str, label_col: int = 3):
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if data.shape[1] <= label_col:
        raise ValueError(f"{path}: only {data.shape[1]} columns, label column is {label_col}")
    xyz = data[:, :3].astype(np.float32)
    raw = data[:, label_col].astype(np.int64)
    label = np.where(raw <= 0, 0, np.where(raw == 1, 1, 2)).astype(np.uint8)
    return xyz, label


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="root folder with one sub-folder per plant")
    p.add_argument("--out", required=True, help="output folder (becomes the training --data root)")
    p.add_argument("--label-col", type=int, default=3)
    p.add_argument("--plants", nargs="*", default=None,
                   help="plant folder names to use (default: every sub-folder containing *_a.txt)")
    p.add_argument("--val-plants", nargs="*", default=[], help="plants held out for validation")
    p.add_argument("--test-plants", nargs="*", default=[], help="plants held out for testing")
    args = p.parse_args(argv)

    plants = args.plants
    if not plants:
        plants = sorted(
            d for d in os.listdir(args.raw)
            if os.path.isdir(os.path.join(args.raw, d))
            and any(f.endswith(POINTCLOUD_SUFFIX) for f in os.listdir(os.path.join(args.raw, d)))
        )
    if not plants:
        print(f"No plant folders with *{POINTCLOUD_SUFFIX} files found in {args.raw}", file=sys.stderr)
        return 1

    held = set(args.val_plants) | set(args.test_plants)
    unknown = held - set(plants)
    if unknown:
        print(f"Unknown held-out plants: {sorted(unknown)} (available: {plants})", file=sys.stderr)
        return 1
    if set(args.val_plants) & set(args.test_plants):
        print("A plant cannot be in both val and test", file=sys.stderr)
        return 1

    splits = {"train": [], "val": [], "test": []}
    hist = np.zeros(3, dtype=np.int64)
    for plant in plants:
        src_dir = os.path.join(args.raw, plant)
        dst_dir = os.path.join(args.out, plant)
        os.makedirs(dst_dir, exist_ok=True)
        split = "val" if plant in args.val_plants else "test" if plant in args.test_plants else "train"
        files = sorted(f for f in os.listdir(src_dir) if f.endswith(POINTCLOUD_SUFFIX))
        for f in files:
            xyz, label = read_annotated_txt(os.path.join(src_dir, f), args.label_col)
            name = f[: -len(POINTCLOUD_SUFFIX)] + ".npz"
            np.savez_compressed(os.path.join(dst_dir, name), xyz=xyz, label=label)
            hist += np.bincount(label, minlength=3)
            splits[split].append(f"{plant}/{name}")
        print(f"{plant}: {len(files)} clouds -> {split}")

    split_dir = os.path.join(args.out, "splits")
    os.makedirs(split_dir, exist_ok=True)
    splits["trainval"] = splits["train"] + splits["val"]
    for name, items in splits.items():
        with open(os.path.join(split_dir, f"Pheno4D_{name}.txt"), "w") as fh:
            fh.write("\n".join(items) + ("\n" if items else ""))

    total = hist.sum()
    print("label histogram (soil, stem, leaf):", hist.tolist(),
          [f"{100.0 * h / total:.1f}%" for h in hist])
    print({k: len(v) for k, v in splits.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
