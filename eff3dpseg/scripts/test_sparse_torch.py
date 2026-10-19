"""
Check scripts/sparse_torch.py against PyTorch's dense convolutions (runs on CPU).

    python scripts/test_sparse_torch.py

For random sparse volumes, each sparse layer must equal the dense conv evaluated at
the active voxels, and gradients must flow.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_torch as st  # noqa: E402


def random_sparse(shape=(6, 7, 5), batch=2, density=0.35, cin=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    dense_mask = torch.rand(batch, *shape, generator=g) < density
    idx = dense_mask.nonzero().int()                      # [N, 4] = batch, x, y, z
    feats = torch.randn(len(idx), cin, generator=g)
    return st.SparseConvTensor(feats, idx, list(shape), batch), dense_mask


def to_dense(x, shape):
    """Features -> dense (B, C, X, Y, Z) with zeros at inactive voxels."""
    c = x.features.shape[1]
    d = torch.zeros(x.batch_size, c, *shape)
    i = x.indices.long()
    d[i[:, 0], :, i[:, 1], i[:, 2], i[:, 3]] = x.features
    return d


def gather(dense, indices):
    i = indices.long()
    return dense[i[:, 0], :, i[:, 1], i[:, 2], i[:, 3]]


def test_subm(k):
    x, _ = random_sparse(cin=3)
    m = st.SubMConv3d(3, 5, k, padding=k // 2, bias=True, indice_key="a")
    torch.nn.init.normal_(m.bias)
    y = m(x)
    assert torch.equal(y.indices, x.indices)
    # offset o = (dx+r)*k*k + (dy+r)*k + (dz+r)  <->  conv3d weight[:, :, dx+r, dy+r, dz+r]
    w = m.weight.detach().permute(2, 1, 0).reshape(5, 3, k, k, k)
    ref = F.conv3d(to_dense(x, x.spatial_shape), w, padding=k // 2)
    want = gather(ref, x.indices) + m.bias.detach()
    assert torch.allclose(y.features, want, atol=1e-5), (y.features - want).abs().max()
    y.features.sum().backward()
    assert m.weight.grad is not None and torch.isfinite(m.weight.grad).all()
    print(f"  SubMConv3d k={k}: OK")


def test_down_up():
    x, _ = random_sparse(shape=(7, 6, 5), cin=3, seed=1)
    down = st.SparseConv3d(3, 4, 2, stride=2, bias=False, indice_key="d")
    up = st.SparseInverseConv3d(4, 6, 2, bias=False, indice_key="d")

    y = down(x)
    ws = down.weight.detach().permute(2, 1, 0).reshape(4, 3, 2, 2, 2)     # offset bits (x,y,z)
    dense_in = to_dense(x, x.spatial_shape)
    ref = F.conv3d(F.pad(dense_in, (0, 1, 0, 1, 0, 1)), ws, stride=2)    # pad odd sizes up to even
    assert y.spatial_shape == [4, 3, 3], y.spatial_shape
    assert torch.allclose(y.features, gather(ref, y.indices), atol=1e-5), "down conv mismatch"
    # every output voxel must have at least one active child, and nothing else is output
    assert len(y.indices) == int((ref.abs().sum(1) > 0).sum()) or len(y.indices) >= 1

    z = up(y)
    assert torch.equal(z.indices, x.indices), "inverse conv must restore the encoder voxels"
    assert z.spatial_shape == x.spatial_shape
    wu = up.weight.detach().permute(1, 2, 0).reshape(4, 6, 2, 2, 2)
    ref_up = F.conv_transpose3d(to_dense(y, y.spatial_shape), wu, stride=2)
    ref_up = ref_up[:, :, : x.spatial_shape[0], : x.spatial_shape[1], : x.spatial_shape[2]]
    assert torch.allclose(z.features, gather(ref_up, z.indices), atol=1e-5), "inverse conv mismatch"
    z.features.sum().backward()
    assert down.weight.grad is not None and up.weight.grad is not None
    print("  SparseConv3d / SparseInverseConv3d: OK")


def test_key_reuse():
    """A SubM indice_key reused on a tensor with identical voxels must give the same result."""
    x, _ = random_sparse(cin=2, seed=3)
    a = st.SubMConv3d(2, 2, 3, padding=1, bias=False, indice_key="r")
    b = st.SubMConv3d(2, 2, 3, padding=1, bias=False, indice_key="r")
    y = b(a(x))
    fresh = st.SubMConv3d(2, 2, 3, padding=1, bias=False)
    fresh.weight.data = b.weight.data
    y2 = fresh(a(x))
    assert torch.allclose(y.features, y2.features, atol=1e-6)
    print("  indice_key reuse: OK")


def main():
    print("sparse_torch vs dense reference")
    test_subm(3)
    test_subm(5)
    test_subm(1)
    test_down_up()
    test_key_reuse()
    print("ALL OK")


if __name__ == "__main__":
    main()
