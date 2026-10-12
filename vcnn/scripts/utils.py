"""Utility helpers: weight init, dice loss, and small visualization tools."""

import numpy as np
import torch
from torch.nn import init


def weights_init(net):
    """Kaiming init for Conv3d/ConvTranspose3d weights, small bias."""
    for m in net.modules():
        if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
            init.kaiming_normal_(m.weight)
            init.constant_(m.bias, 0.01)


def dice_loss(y_conv, y_true):
    """Dice loss computed on the **positive** class to handle class imbalance.

    Arguments:
        y_true: [N] or [N, 1]  long tensor of 0/1 labels.
        y_conv: [N, 2]         float tensor; column 1 is the positive-class prob.
    """
    y_conv = y_conv[:, 1]
    y_true = y_true.float()
    intersection = torch.sum(y_conv * y_true, 0)
    union = torch.sum(y_conv * y_conv, 0) + torch.sum(y_true * y_true, 0)
    dice = 2.0 * intersection / union
    return 1 - torch.clamp(dice, 0.0, 1.0 - 1e-7)


def show_center_slices(im_3d, indices=None):
    """Display the three orthogonal center slices of a 3D volume (notebook use)."""
    import matplotlib.pyplot as plt
    plt.rcParams["image.cmap"] = "gray"

    if indices is None:
        indices = np.array(im_3d.shape) // 2
    assert len(indices) == 3, \
        "Expect a 3-D array, but received a %d-D array." % len(indices)

    x_th, y_th, z_th = indices
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(im_3d[x_th, :, :])
    axes[1].imshow(im_3d[:, y_th, :])
    axes[2].imshow(im_3d[:, :, z_th])
    plt.suptitle("Center slices of the 3-D volume")
    return fig
