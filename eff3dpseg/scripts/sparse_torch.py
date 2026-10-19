"""
Pure-PyTorch sparse 3D convolutions: a drop-in subset of ``spconv.pytorch``.

Used when spconv has no working build for the machine (e.g. CUDA 13 Colab). Only
what models.py needs is implemented:

    SparseConvTensor(features, indices, spatial_shape, batch_size)
    SubMConv3d(cin, cout, k, padding, bias, indice_key)     odd k, same voxels in/out
    SparseConv3d(cin, cout, 2, stride=2, bias, indice_key)  halves the resolution
    SparseInverseConv3d(cin, cout, 2, bias, indice_key)     undoes a SparseConv3d
    SparseSequential

Semantics match spconv (zero outside active voxels, indices = [batch, x, y, z]).
It uses a "rulebook": for every kernel offset, the (input row, output row) pairs
that exist, gathered with sort + searchsorted, then ``index_add_``. It is slower
and uses more memory than spconv's fused CUDA kernels, but works on any torch/CUDA
(and on CPU), with autograd and AMP.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, indice_dict=None):
        self.features = features
        self.indices = indices
        self.spatial_shape = [int(s) for s in spatial_shape]
        self.batch_size = int(batch_size)
        # Rulebooks shared by every tensor of one forward pass (mutable on purpose).
        self.indice_dict = {} if indice_dict is None else indice_dict

    def replace_feature(self, features):
        return SparseConvTensor(features, self.indices, self.spatial_shape,
                                self.batch_size, self.indice_dict)


def _encode(batch, xyz, shape):
    X, Y, Z = shape
    return ((batch * X + xyz[:, 0]) * Y + xyz[:, 1]) * Z + xyz[:, 2]


def _subm_rulebook(indices, shape, k):
    """For each of the k^3 offsets: (input_rows, output_rows) of existing neighbours."""
    idx = indices.long()
    batch, xyz = idx[:, 0], idx[:, 1:]
    n = idx.shape[0]
    bound = torch.tensor(shape, device=idx.device)
    sorted_keys, perm = torch.sort(_encode(batch, xyz, shape))
    r = k // 2
    rules = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                nb = xyz + torch.tensor([dx, dy, dz], device=idx.device)
                inside = ((nb >= 0) & (nb < bound)).all(1)
                q = _encode(batch, nb, shape)
                pos = torch.searchsorted(sorted_keys, q).clamp_(max=n - 1)
                hit = inside & (sorted_keys[pos] == q)
                out_rows = hit.nonzero(as_tuple=True)[0]
                rules.append((perm[pos[out_rows]], out_rows))
    return rules


class SubMConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True, indice_key=None):
        super().__init__()
        assert kernel_size % 2 == 1, "submanifold conv needs an odd kernel"
        assert padding == kernel_size // 2, "submanifold conv is 'same' padded"
        self.k, self.key, self.cout = kernel_size, indice_key, out_channels
        K = kernel_size ** 3
        self.weight = nn.Parameter(torch.empty(K, in_channels, out_channels))
        nn.init.normal_(self.weight, 0.0, math.sqrt(2.0 / (in_channels * K)))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        rules = None
        cache_key = (self.key, self.k) if self.key is not None else None
        if cache_key is not None:
            rules = x.indice_dict.get(cache_key)
        if rules is None:
            rules = _subm_rulebook(x.indices, x.spatial_shape, self.k)
            if cache_key is not None:
                x.indice_dict[cache_key] = rules
        n = x.features.shape[0]
        out = None
        for o, (src, dst) in enumerate(rules):
            if src.numel() == 0:
                continue
            prod = x.features[src] @ self.weight[o]
            if out is None:
                out = prod.new_zeros(n, self.cout)
            out.index_add_(0, dst, prod)
        if out is None:
            out = x.features.new_zeros(n, self.cout)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return SparseConvTensor(out, x.indices, x.spatial_shape, x.batch_size, x.indice_dict)


class SparseConv3d(nn.Module):
    """Kernel 2, stride 2 (the only configuration the model uses)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 indice_key=None):
        super().__init__()
        if kernel_size != 2 or stride != 2 or padding != 0:
            raise NotImplementedError("sparse_torch.SparseConv3d supports kernel=2, stride=2 only")
        self.key, self.cout = indice_key, out_channels
        self.weight = nn.Parameter(torch.empty(8, in_channels, out_channels))
        nn.init.normal_(self.weight, 0.0, math.sqrt(2.0 / (in_channels * 8)))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        idx = x.indices.long()
        batch, xyz = idx[:, 0], idx[:, 1:]
        half = torch.div(xyz, 2, rounding_mode="floor")
        parity = xyz - 2 * half
        offset = parity[:, 0] * 4 + parity[:, 1] * 2 + parity[:, 2]
        out_idx, inverse = torch.unique(torch.cat([batch[:, None], half], 1), dim=0, return_inverse=True)
        n_out = out_idx.shape[0]
        if self.key is not None:
            x.indice_dict[self.key] = {
                "in_indices": x.indices, "in_shape": x.spatial_shape,
                "inverse": inverse, "offset": offset, "n_out": n_out,
            }
        out = None
        for o in range(8):
            sel = (offset == o).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            prod = x.features[sel] @ self.weight[o]
            if out is None:
                out = prod.new_zeros(n_out, self.cout)
            out.index_add_(0, inverse[sel], prod)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        shape = [(s + 1) // 2 for s in x.spatial_shape]
        return SparseConvTensor(out, out_idx.int(), shape, x.batch_size, x.indice_dict)


class SparseInverseConv3d(nn.Module):
    """Inverse of a SparseConv3d(k=2, s=2) with the same indice_key: restores its input voxels."""

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, indice_key=None):
        super().__init__()
        if kernel_size != 2 or indice_key is None:
            raise NotImplementedError("sparse_torch.SparseInverseConv3d needs kernel=2 and an indice_key")
        self.key, self.cout = indice_key, out_channels
        self.weight = nn.Parameter(torch.empty(8, in_channels, out_channels))
        nn.init.normal_(self.weight, 0.0, math.sqrt(2.0 / (in_channels * 8)))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        rb = x.indice_dict[self.key]
        assert x.features.shape[0] == rb["n_out"], "inverse conv input does not match its downsample"
        inverse, offset = rb["inverse"], rb["offset"]
        n = inverse.numel()
        out = None
        for o in range(8):
            sel = (offset == o).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            prod = x.features[inverse[sel]] @ self.weight[o]
            if out is None:
                out = prod.new_zeros(n, self.cout)
            out.index_copy_(0, sel, prod)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return SparseConvTensor(out, rb["in_indices"], rb["in_shape"], x.batch_size, x.indice_dict)


# Modules here take and return SparseConvTensor, so a plain Sequential composes them.
SparseSequential = nn.Sequential
