"""
VCNN model definition (voxel-based 3D U-Net) + discriminative instance loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class conv3d(nn.Module):
    """conv3d -> ReLU (no norm layer, per the paper's note that it vanishes)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, activation_func=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = activation_func()

    def forward(self, x):
        return self.relu(self.conv(x))


class conv3d_x3(nn.Module):
    """Three serial convs with a residual (skip) connection.

        inputs --> conv1 --> conv2 --> conv3 --> outputs
                    └──────────── add ───────────┘
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv_1 = conv3d(in_channels, out_channels, kernel_size)
        self.conv_2 = conv3d(out_channels, out_channels, kernel_size)
        self.conv_3 = conv3d(out_channels, out_channels, kernel_size)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        return z_1 + z_3


class deconv3d_x3(nn.Module):
    """Upsample `rhs`, fuse with skip connection `lhs`, then 3x residual convs."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 activation_func=nn.ReLU):
        super().__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, kernel_size, stride)
        self.lhs_conv = conv3d(out_channels // 2, out_channels, kernel_size)
        self.conv_x3 = conv3d_x3(out_channels, out_channels, kernel_size)

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)                 # upsample decoder path
        lhs_conv = self.lhs_conv(lhs)         # match channels of skip connection
        rhs_add = crop(rhs_up, lhs_conv) + lhs_conv   # element-wise fusion
        return self.conv_x3(rhs_add)


def crop(large, small):
    """Center-crop `large` to the spatial size of `small`.

    Shapes are [batch, channels, depth, height, width].
    """
    l, s = large.size(), small.size()
    off_d = (l[2] - s[2]) // 2
    off_h = (l[3] - s[3]) // 2
    off_w = (l[4] - s[4]) // 2
    return large[..., off_d:off_d + s[2], off_h:off_h + s[3], off_w:off_w + s[4]]


def conv3d_as_pool(in_channels, out_channels, kernel_size=3, stride=2,
                   activation_func=nn.ReLU):
    """Strided conv used as a learnable downsampling (pooling) layer."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=1),
        activation_func(),
    )


def deconv3d_as_up(in_channels, out_channels, kernel_size=3, stride=2,
                   activation_func=nn.ReLU):
    """Transposed conv used as upsampling."""
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
        activation_func(),
    )


# --------------------------------------------------------------------------- #
# Output heads
# --------------------------------------------------------------------------- #
class softmax_out(nn.Module):
    """Semantic-classification head: 1x1 conv producing per-voxel class logits."""

    def __init__(self, in_channels, out_channels, criterion):
        super().__init__()
        self._K = out_channels
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)
        if criterion == "nll":
            self.softmax = F.log_softmax
        else:
            assert criterion == "dice", \
                "Expect `dice` (dice loss) or `nll` (negative log likelihood loss)."
            self.softmax = F.softmax

    def forward(self, x):
        y_conv = self.conv_2(self.conv_1(x))
        # channel axis -> last dim for softmax
        y_perm = y_conv.permute(0, 2, 3, 4, 1).contiguous()
        y_flat = y_perm.view(-1, self._K)
        return self.softmax(y_flat, dim=1), y_conv


class conv12(nn.Module):
    """Segmentation-embedding head: 3x3 then 1x1 conv -> per-voxel feature."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        y_conv = self.conv_2(self.conv_1(x))
        return y_conv.permute(0, 2, 3, 4, 1).contiguous()


# --------------------------------------------------------------------------- #
# Discriminative loss for instance segmentation (De Brabandere et al., 2017)
# --------------------------------------------------------------------------- #
def disc_loss(fts, y, conf):
    """Discriminative embedding loss.

    Arguments:
        fts: [N, C] per-point embedding features.
        y:   [N]    ground-truth instance ids (0..K-1; 0 treated as background).
        conf: dict with keys 'c0', 'c1', 'abcd'.

    Returns:
        A scalar loss combining:
          * variance term  – pull points toward their instance center,
          * distance term  – push different instance centers apart,
          * regularizer    – keep centers near a target radius.
    """
    N, C = fts.size()
    y = y.contiguous().view(-1)

    num = int(y.max().item()) + 1 if y.numel() > 0 else 0

    rs_var, rs_dist, rs_center = [], [], []
    for k in range(num):
        mask = y.eq(k).view(-1, 1).expand_as(fts)
        val = fts.masked_select(mask).view(-1, C)
        if val.shape[0] == 0:
            continue
        cur_center = val.mean(0)
        rs_center.append(cur_center)

        dist = torch.norm(val - cur_center.expand_as(val), 2, 1)
        dist = dist - conf["c0"]
        dist = torch.clamp(dist, min=1e-7)
        rs_var.append((dist * dist).mean())

    # inter-cluster distance (ignore background index 0)
    # Use the actual number of populated centers so empty instance ids
    # (skipped above) can't cause an out-of-range index.
    n_centers = len(rs_center)
    for i in range(1, n_centers):
        for j in range(n_centers):
            if i >= j:
                continue
            val = 2 * conf["c1"] - (rs_center[i] - rs_center[j]).norm(p=2)
            val = torch.clamp(val, min=1e-7)
            rs_dist.append(val * val)

    rs_center_ = torch.stack(rs_center).view(-1, C)
    rs_center_norm = torch.norm(rs_center_, 2, 1)          # per-instance center norms
    rs_center_norm = torch.clamp(rs_center_norm, min=1e-7)
    # Regularizer: pull all instance centers toward a target radius (6 * c0).
    rs_add = torch.norm(rs_center_norm - 6 * conf["c0"], 2)
    rs_add = torch.clamp(rs_add, min=1e-7)

    a, b, c, d = conf["abcd"]
    loss = (a * torch.stack(rs_var).mean()
            + b * (torch.stack(rs_dist).mean() if rs_dist else fts.new_zeros(()))
            + d * rs_add.mean())
    return loss


# --------------------------------------------------------------------------- #
# Full network
# --------------------------------------------------------------------------- #
class VCNN(nn.Module):
    """3D U-Net with two heads: classification (K classes) + instance embedding."""

    def __init__(self, K, criterion):
        super().__init__()
        # ---- encoder ---- #
        self.conv_1 = conv3d_x3(1, 16)
        self.pool_1 = conv3d_as_pool(16, 32)
        self.conv_2 = conv3d_x3(32, 32)
        self.pool_2 = conv3d_as_pool(32, 64)
        self.conv_3 = conv3d_x3(64, 64)
        self.pool_3 = conv3d_as_pool(64, 128)
        self.conv_4 = conv3d_x3(128, 128)
        self.pool_4 = conv3d_as_pool(128, 256)

        # ---- bottleneck ---- #
        self.bottom = conv3d_x3(256, 256)

        # ---- decoder ---- #
        self.deconv_4 = deconv3d_x3(256, 256)
        self.deconv_3 = deconv3d_x3(256, 128)
        self.deconv_2 = deconv3d_x3(128, 64)
        self.deconv_1 = deconv3d_x3(64, 32)

        # ---- heads ---- #
        self.out = softmax_out(32, K, criterion)
        self.out2 = conv12(32 + K, 32)
        self.K = K

    def forward(self, x):
        # encoder
        conv_1 = self.conv_1(x)
        pool = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool)
        pool = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool)
        pool = self.pool_3(conv_3)
        conv_4 = self.conv_4(pool)
        pool = self.pool_4(conv_4)

        # bottleneck
        bottom = self.bottom(pool)

        # decoder with skip connections
        deconv = self.deconv_4(conv_4, bottom)
        deconv = self.deconv_3(conv_3, deconv)
        deconv = self.deconv_2(conv_2, deconv)
        deconv = self.deconv_1(conv_1, deconv)

        # classification head
        rs_cls, y_conv = self.out(deconv)

        # segmentation-embedding head: fuse class features + decoder features
        fts = torch.cat([y_conv, deconv], dim=1)     # channels == 32 + K
        assert fts.size(1) == 32 + self.K
        rs_fts = self.out2(fts)

        rs_fts = rs_fts.view(rs_cls.size()[0], -1)
        return rs_cls, rs_fts

