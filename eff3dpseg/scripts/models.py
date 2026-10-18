"""
Res16UNet34C re-implemented with spconv (v2) instead of MinkowskiEngine.

Mapping from the original models/res16unet.py:
    ME conv k=conv1_kernel_size, stride 1   -> SubMConv3d
    ME conv k=2, stride 2 (downsample)      -> SparseConv3d (indice_key 'downN')
    ME conv_tr k=2, stride 2 (upsample)     -> SparseInverseConv3d (same indice_key)
    ME conv k=3 inside BasicBlock           -> SubMConv3d (indice_key per level)
    me.cat                                  -> channel concat of .features

SparseInverseConv3d reuses the downsample's index pairs, so the decoder voxels
are exactly the encoder voxels of that level and the skip concatenation lines up.
Layer/plane configuration is unchanged: LAYERS=(2,3,4,6,2,2,2,2),
PLANES=(32,64,128,256,256,128,96,96), INIT_DIM=32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import spconv.pytorch as spconv


def _bn(c, momentum):
    return nn.BatchNorm1d(c, momentum=momentum)


class _Norm(nn.Module):
    """BatchNorm (+ optional ReLU) on the features of a SparseConvTensor."""

    def __init__(self, c, momentum, relu=True):
        super().__init__()
        self.bn = _bn(c, momentum)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        f = self.bn(x.features)
        if self.relu is not None:
            f = self.relu(f)
        return x.replace_feature(f)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, key, momentum):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(inplanes, planes, 3, padding=1, bias=False, indice_key=key)
        self.bn1 = _bn(planes, momentum)
        self.conv2 = spconv.SubMConv3d(planes, planes, 3, padding=1, bias=False, indice_key=key)
        self.bn2 = _bn(planes, momentum)
        self.relu = nn.ReLU(inplace=True)
        # 1x1 projection on the features when the channel count changes
        self.down = None
        if inplanes != planes:
            self.down = nn.Sequential(nn.Linear(inplanes, planes, bias=False), _bn(planes, momentum))

    def forward(self, x):
        identity = x.features if self.down is None else self.down(x.features)
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.relu(self.bn2(out.features) + identity))
        return out


def _make_layer(inplanes, planes, blocks, key, momentum):
    layers = [BasicBlock(inplanes, planes, key, momentum)]
    layers += [BasicBlock(planes, planes, key, momentum) for _ in range(blocks - 1)]
    return spconv.SparseSequential(*layers)


class Res16UNet34C(nn.Module):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    INIT_DIM = 32

    def __init__(self, in_channels, out_channels, conv1_kernel_size=5, bn_momentum=0.02):
        super().__init__()
        P, L, m, D = self.PLANES, self.LAYERS, bn_momentum, self.INIT_DIM
        k1 = conv1_kernel_size

        def down(c, key):
            return spconv.SparseConv3d(c, c, 2, stride=2, bias=False, indice_key=key)

        def up(cin, cout, key):
            return spconv.SparseInverseConv3d(cin, cout, 2, bias=False, indice_key=key)

        # encoder
        self.conv0 = spconv.SubMConv3d(in_channels, D, k1, padding=k1 // 2, bias=False, indice_key="subm0")
        self.n0 = _Norm(D, m)
        self.conv1, self.n1 = down(D, "down1"), _Norm(D, m)
        self.block1 = _make_layer(D, P[0], L[0], "subm1", m)
        self.conv2, self.n2 = down(P[0], "down2"), _Norm(P[0], m)
        self.block2 = _make_layer(P[0], P[1], L[1], "subm2", m)
        self.conv3, self.n3 = down(P[1], "down3"), _Norm(P[1], m)
        self.block3 = _make_layer(P[1], P[2], L[2], "subm3", m)
        self.conv4, self.n4 = down(P[2], "down4"), _Norm(P[2], m)
        self.block4 = _make_layer(P[2], P[3], L[3], "subm4", m)

        # decoder: each convtr undoes the matching downsample; keys 'subm3'.. are
        # reused by the decoder blocks because their voxels are the encoder's.
        self.tr4, self.ntr4 = up(P[3], P[4], "down4"), _Norm(P[4], m)
        self.block5 = _make_layer(P[4] + P[2], P[4], L[4], "subm3", m)
        self.tr5, self.ntr5 = up(P[4], P[5], "down3"), _Norm(P[5], m)
        self.block6 = _make_layer(P[5] + P[1], P[5], L[5], "subm2", m)
        self.tr6, self.ntr6 = up(P[5], P[6], "down2"), _Norm(P[6], m)
        self.block7 = _make_layer(P[6] + P[0], P[6], L[6], "subm1", m)
        self.tr7, self.ntr7 = up(P[6], P[7], "down1"), _Norm(P[7], m)
        # own key: 'subm0' was built for the k=conv1_kernel_size first conv, spconv
        # requires the same kernel size when an indice_key is reused.
        self.block8 = _make_layer(P[7] + D, P[7], L[7], "subm0_k3", m)

        self.final = spconv.SubMConv3d(P[7], out_channels, 1, bias=True, indice_key="final")

    @staticmethod
    def _cat(a, b):
        return a.replace_feature(torch.cat([a.features, b.features], dim=1))

    def forward(self, x):
        out_p1 = self.n0(self.conv0(x))
        out = self.n1(self.conv1(out_p1))
        out_b1 = self.block1(out)
        out = self.n2(self.conv2(out_b1))
        out_b2 = self.block2(out)
        out = self.n3(self.conv3(out_b2))
        out_b3 = self.block3(out)
        out = self.n4(self.conv4(out_b3))
        out = self.block4(out)

        out = self.block5(self._cat(self.ntr4(self.tr4(out)), out_b3))
        out = self.block6(self._cat(self.ntr5(self.tr5(out)), out_b2))
        out = self.block7(self._cat(self.ntr6(self.tr6(out)), out_b1))
        out = self.block8(self._cat(self.ntr7(self.tr7(out)), out_p1))
        return self.final(out)


MODELS = {"Res16UNet34C": Res16UNet34C}


def build_model(name, in_channels, out_channels, conv1_kernel_size=5, bn_momentum=0.02):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](in_channels, out_channels, conv1_kernel_size, bn_momentum)


def to_sparse(batch, device):
    """Collated batch -> SparseConvTensor (indices int32 [batch, x, y, z])."""
    return spconv.SparseConvTensor(
        batch["feats"].to(device),
        batch["coords"].to(device),
        batch["spatial_shape"],
        batch["batch_size"],
    )
