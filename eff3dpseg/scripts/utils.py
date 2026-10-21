"""Metrics, seeding and a dependency-free PLY writer."""

from __future__ import annotations

import random

import numpy as np

# RGB per training class (stem, leaf) for visualisation; grey for ignored points.
CLASS_COLORS = np.array([[174, 199, 232], [152, 223, 138]], dtype=np.uint8)
IGNORE_COLOR = np.array([128, 128, 128], dtype=np.uint8)


def fast_hist(pred: np.ndarray, label: np.ndarray, n: int) -> np.ndarray:
    """Confusion matrix (rows = ground truth). Labels outside [0, n) are skipped."""
    k = (label >= 0) & (label < n) & (pred >= 0) & (pred < n)
    return np.bincount(n * label[k].astype(int) + pred[k].astype(int), minlength=n * n).reshape(n, n)


def per_class_iu(hist: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def prf_iou(hist: np.ndarray) -> np.ndarray:
    """Per-class [precision, recall, F1, IoU] (in %), shape (n_classes, 4), from a confusion matrix.

    Same definitions as the Eff-3DPSeg paper (eqs. 6-9): rows of ``hist`` are ground truth.
    A class with no support gives NaN (excluded from the mean).
    """
    tp = np.diag(hist).astype(np.float64)
    fn = hist.sum(1) - tp
    fp = hist.sum(0) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        iou = tp / (tp + fp + fn)
    return 100.0 * np.stack([precision, recall, f1, iou], axis=1)


def format_metrics(table: np.ndarray, class_names) -> str:
    """Paper-style table: one row per metric, one column per class plus the class mean."""
    head = "            " + "".join(f"{c:>9}" for c in class_names) + f"{'mean':>9}"
    lines = [head]
    for k, name in enumerate(("Precision", "Recall", "F1-score", "IoU")):
        vals = table[:, k]
        lines.append(f"{name:<12}" + "".join(f"{v:9.1f}" for v in vals) + f"{np.nanmean(vals):9.1f}")
    return "\n".join(lines)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_ply(path: str, xyz: np.ndarray, pred: np.ndarray, label: np.ndarray | None = None) -> None:
    """Binary PLY with x y z, red green blue (prediction colour) and the predicted class."""
    colors = np.tile(IGNORE_COLOR, (len(xyz), 1))
    ok = (pred >= 0) & (pred < len(CLASS_COLORS))
    colors[ok] = CLASS_COLORS[pred[ok]]
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("pred", "u1")]
    rec = np.empty(len(xyz), dtype=dtype)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    rec["pred"] = pred.astype(np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar pred\n"
        "end_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rec.tobytes())
