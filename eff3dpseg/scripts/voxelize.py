"""
Numpy-only voxelization + augmentation (replaces MinkowskiEngine's sparse_quantize,
lib/voxelizer.py and lib/transforms.py of the original repo).
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage
import scipy.interpolate

IGNORE = 255
# raw Pheno4D id -> training id (soil is ignored; stem=0, leaf=1)
LABEL_MAP = np.array([IGNORE, 0, 1], dtype=np.int64)


def _rot(axis: int, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    m = np.eye(3)
    m[i, i], m[j, j] = c, c
    # sign convention: standard right-handed rotation about each axis
    if axis == 1:
        m[i, j], m[j, i] = s, -s
    else:
        m[i, j], m[j, i] = -s, s
    return m


def random_rotation(bounds, rng: np.random.Generator) -> np.ndarray:
    """Compose per-axis rotations in random order (as the original Voxelizer)."""
    mats = []
    for axis, b in enumerate(bounds):
        theta = 0.0 if b is None else rng.uniform(*b)
        mats.append(_rot(axis, theta))
    order = rng.permutation(3)
    return mats[order[0]] @ mats[order[1]] @ mats[order[2]]


def elastic_distortion(coords: np.ndarray, granularity: float, magnitude: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Smooth random displacement field, evaluated at the points. Units: voxels."""
    blur = [np.ones((3, 1, 1, 1), np.float32) / 3,
            np.ones((1, 3, 1, 1), np.float32) / 3,
            np.ones((1, 1, 3, 1), np.float32) / 3]
    cmin = coords.min(0)
    noise_dim = ((coords - cmin).max(0) // granularity).astype(int) + 3
    noise = rng.standard_normal((*noise_dim, 3)).astype(np.float32)
    for _ in range(2):
        for b in blur:
            noise = scipy.ndimage.convolve(noise, b, mode="constant", cval=0)
    ax = [np.linspace(lo, hi, d) for lo, hi, d in
          zip(cmin - granularity, cmin + granularity * (noise_dim - 2), noise_dim)]
    interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=False, fill_value=0)
    return coords + interp(coords) * magnitude


def quantize(vox: np.ndarray):
    """Floor to integer voxels, shift to origin, deduplicate.

    Returns (coords int32 (M,3), first_index (M,), inverse (N,)) where
    ``coords[inverse] == floor(vox) - min`` for every input point.
    """
    ijk = np.floor(vox).astype(np.int64)
    ijk -= ijk.min(0)
    coords, first, inverse = np.unique(ijk, axis=0, return_index=True, return_inverse=True)
    return coords.astype(np.int32), first, inverse.reshape(-1)


def voxel_labels(point_labels: np.ndarray, inverse: np.ndarray, n_vox: int) -> np.ndarray:
    """Per-voxel label; voxels whose points disagree (or hit soil) become IGNORE."""
    lo = np.full(n_vox, 255, dtype=np.int64)
    hi = np.zeros(n_vox, dtype=np.int64)
    np.minimum.at(lo, inverse, point_labels)
    np.maximum.at(hi, inverse, point_labels)
    return np.where(lo == hi, lo, IGNORE)


def map_labels(raw: np.ndarray) -> np.ndarray:
    """Raw ids {0,1,2} -> training ids {255,0,1}; anything else -> IGNORE."""
    raw = raw.astype(np.int64)
    out = np.full(raw.shape, IGNORE, dtype=np.int64)
    ok = (raw >= 0) & (raw < len(LABEL_MAP))
    out[ok] = LABEL_MAP[raw[ok]]
    return out


def make_features(coords: np.ndarray, mode: str) -> np.ndarray:
    if mode == "ones":
        return np.ones((len(coords), 1), dtype=np.float32)
    if mode == "xyz":
        c = coords.astype(np.float32)
        c = c - c.mean(0)
        return c / max(float(np.abs(c).max()), 1.0)
    raise ValueError(f"unknown feature_mode {mode!r}")


def voxelize_sample(xyz: np.ndarray, raw_label: np.ndarray, conf, train: bool,
                    rng: np.random.Generator):
    """Full sample pipeline.

    Returns a dict with voxel ``coords`` (M,3 int32), ``feats`` (M,C float32),
    ``labels`` (M,) int64, and for evaluation ``inverse`` (N,) and ``point_labels``.
    """
    xyz = xyz.astype(np.float64)
    label = map_labels(raw_label)

    if train:
        if conf.point_dropout and rng.random() < conf.point_dropout:
            keep = rng.choice(len(xyz), int(len(xyz) * (1 - conf.point_dropout)), replace=False)
            xyz, label = xyz[keep], label[keep]
        centre = xyz.mean(0)
        xyz = (xyz - centre) @ random_rotation(conf.rotation_aug, rng).T
        scale = rng.uniform(*conf.scale_aug) if conf.scale_aug else 1.0
    else:
        scale = 1.0

    vox = xyz * (scale / conf.voxel_size)

    if train:
        if conf.elastic_params:
            g, m = conf.elastic_params[rng.integers(len(conf.elastic_params))]
            vox = elastic_distortion(vox, g, m, rng)
        for axis in (0, 1):                      # horizontal flips, z is upright
            if rng.random() < 0.5:
                vox[:, axis] = vox[:, axis].max() - vox[:, axis]

    coords, _, inverse = quantize(vox)
    out = {
        "coords": coords,
        "feats": make_features(coords, conf.feature_mode),
        "labels": voxel_labels(label, inverse, len(coords)),
    }
    if not train:
        out["inverse"] = inverse
        out["point_labels"] = label
    return out
