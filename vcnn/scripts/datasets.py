"""
Dataset for voxelized maize point clouds stored in PyTables `.h5` files.

Each `.h5` file has a group `/process/` with four arrays (see data/data.md):
    x    : (N, 3)   float  voxel coordinates (all samples concatenated)
    y1   : (N,)     int    classification label  (0 = background, 1 = plant)
    y2   : (N,)     int    instance/segmentation label (0..K-1)
    idxs : (M, 2)   int    (start, end) slice bounds delimiting each sample

Ported / cleaned from the original:
  * Removed the IPython `embed()` breakpoint that would hang on Kaggle/Colab.
  * Implemented the previously-missing `_get_test_data` (validation/inference).
  * Factorized the shared voxelization into `_voxelize`.
  * Fully device-agnostic; returns tensors/numpy only (DataLoader handles device).
"""

import os

import numpy as np
import torch
import torch.utils.data as data
from scipy.sparse import csr_matrix

from h5 import h5f_

np.random.seed(1234567)
torch.manual_seed(1234567)


def rot_c3x3(rpy):
    """Rotation matrix from roll-pitch-yaw angles (columns: R*x => y)."""
    cx, cy, cz = np.cos(rpy)
    sx, sy, sz = np.sin(rpy)
    return np.array([
        [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy,     cy * sx,                cx * cy],
    ])


class PartDataset(data.Dataset):
    def __init__(self, fname, size, batch, rot_limit=None, trans_limit=None,
                 train=True):
        """
        Arguments:
            fname      : path to the `.h5` dataset.
            size       : number of items returned by `__len__` (dataset is virtual).
            batch      : batch size (used to advance the sample iterator).
            rot_limit  : (3,) max rotation (rad) per axis; default [5,5,360] deg.
            trans_limit: (3,) max translation; default zeros.
            train      : True -> random augmentation; False -> deterministic.
        """
        self._fname = fname
        assert os.path.exists(fname), "dataset not found: %s" % fname
        self._xyz = None

        with h5f_(fname, mode="r") as h5f:
            y1 = h5f.get_node("/process/y1").read()
            y2 = h5f.get_node("/process/y2").read()
            idxs = h5f.get_node("/process/idxs").read()

        self._cls = y1
        self._parts = y2
        self._fidx = idxs
        self._size = size
        self._batch = batch
        self._itr = -1
        self._rot_limit = (rot_limit if rot_limit is not None
                           else np.r_[5.0, 5.0, 360.0] * np.pi / 180)
        self._trans_limit = (trans_limit if trans_limit is not None
                             else np.r_[0.0, 0.0, 0.0])
        self._nset = self._fidx.shape[0]
        self._train = train

    # ------------------------------------------------------------------ #
    #  Shared voxelization
    # ------------------------------------------------------------------ #
    def _voxelize(self, cur_xyz, cur_cls, cur_part):
        """Voxelize one plant into a 32^3 occupancy grid.

        Returns:
            gt_x        : (1, nx, ny, nz) float occupancy volume.
            gt_y        : (n_vox, 3) int64 [cls, part, roi] for occupied voxels.
            vox_xyz     : (n_vox, 3) int32 unique voxel coordinates.
            scale       : float normalization scale (31 / max_extent).
            minxyz      : (3,) float coordinate min subtracted (for un-normalizing).
            raw_to_voxel: (N_raw,) int index mapping each raw point -> voxel row.
        """
        minxyz = cur_xyz.min(0)
        norm = cur_xyz - minxyz
        scale = 31.0 / norm.max(0).min()
        xyz_int = (norm * scale).astype("i4")

        # unique voxels + inverse mapping raw point -> unique voxel row
        _val = xyz_int.view("S12").flatten()
        _, keep, inverse = np.unique(_val, return_index=True, return_inverse=True)
        keep = keep.astype("i8")
        inverse = inverse.astype("i8").reshape(-1)

        vox_xyz = xyz_int[keep]
        vox_cls = cur_cls[keep].astype("i4")
        vox_part = cur_part[keep].astype("i4")

        nx, ny, nz = vox_xyz.max(0) + 1
        val = np.ones(vox_xyz.shape[0], dtype="f4")
        idx = (vox_xyz[:, 0] * ny + vox_xyz[:, 1], vox_xyz[:, 2])
        gt_x = csr_matrix((val, idx), shape=(nx * ny, nz))
        gt_x = gt_x.astype("f4").toarray().reshape(1, nx, ny, nz)

        roi = (vox_xyz[:, 0] * ny + vox_xyz[:, 1]) * nz + vox_xyz[:, 2]
        gt_y = np.c_[vox_cls, vox_part, roi].astype("i8")

        return gt_x, gt_y, vox_xyz, scale, minxyz, inverse

    def _get_data(self, i):
        """Training item with random rotation augmentation."""
        ran = np.random.rand(6) * 2 - 1
        rot = ran[:3] * self._rot_limit
        trans = ran[3:] * self._trans_limit

        s = slice(*self._fidx[i])
        cur_xyz = self._xyz[s]
        cur_xyz_o = np.copy(cur_xyz)
        cur_cls = self._cls[s].astype("i4")
        cur_part = self._parts[s].astype("i4")

        cur_xyz = cur_xyz.dot(rot_c3x3(rot)) + trans

        gt_x, gt_y, vox_xyz, scale, _, _ = self._voxelize(cur_xyz, cur_cls, cur_part)
        return gt_x, gt_y, vox_xyz, scale, cur_xyz_o

    def _get_test_data(self, i):
        """Deterministic item for validation / inference (no random augmentation).

        Returns extra fields so predictions can be mapped back to raw points:
            (gt_x, gt_y, vox_xyz, scale, minxyz, cur_xyz_o, cur_cls, cur_part,
             raw_to_voxel)
        """
        s = slice(*self._fidx[i])
        cur_xyz = self._xyz[s]
        cur_xyz_o = np.copy(cur_xyz)
        cur_cls = self._cls[s].astype("i4")
        cur_part = self._parts[s].astype("i4")

        # deterministic: identity rotation, zero translation
        gt_x, gt_y, vox_xyz, scale, minxyz, inverse = self._voxelize(
            cur_xyz, cur_cls, cur_part)

        return (gt_x, gt_y, vox_xyz, scale, minxyz, cur_xyz_o,
                cur_cls, cur_part, inverse)

    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        if index % self._batch == 0:            # advance sample per batch
            self._itr = (self._itr + 1) % self._nset

        if self._xyz is None:                   # lazy-load large x array once
            # NOTE: the demo file stores `x` as a disk-backed PyTables CArray, so
            # we must keep the file handle open for the lifetime of the dataset.
            self._h5f = h5f_(self._fname, mode="r")
            self._xyz = self._h5f.get_node("/process/x")

        if self._train:
            gt_x, gt_y, vox_xyz, scale, cur_xyz_o = self._get_data(self._itr)
            return (torch.from_numpy(gt_x), torch.from_numpy(gt_y),
                    vox_xyz, scale, cur_xyz_o)
        else:
            (gt_x, gt_y, vox_xyz, scale, minxyz, cur_xyz_o,
             cur_cls, cur_part, inverse) = self._get_test_data(self._itr)
            return (torch.from_numpy(gt_x), torch.from_numpy(gt_y),
                    vox_xyz, scale, minxyz, cur_xyz_o,
                    cur_cls, cur_part, inverse)

    def __len__(self):
        return self._size

