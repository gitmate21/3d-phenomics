#!/usr/bin/env python
"""
Train the VCNN (voxel-based CNN) for maize stem/leaf classification + segmentation.

Usage (Kaggle / Colab / local):
    python scripts/train.py --data data/x_y1_y2_idx_33_withoutnoise.h5 \
        --model vcnn_demo --epochs 5
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Allow running as `python scripts/train.py` from anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../vcnn/scripts
_ROOT = os.path.dirname(_HERE)                              # .../vcnn
sys.path.insert(0, _HERE)                                   # for datasets/utils/vcnnmodel
sys.path.insert(0, _ROOT)                                   # for config

from config import conf
from datasets import PartDataset
from utils import weights_init
from vcnnmodel import VCNN, disc_loss

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s | %(message)s",
    level=logging.INFO, datefmt="%m-%d %H:%M")


def build_loaders(data_path, batch_size, threads):
    """Create train and validation DataLoaders from a single `.h5` file.

    The original used the same demo file for both; override `--val` for real data.
    """
    train_ds = PartDataset(data_path, 300, batch_size, train=True)
    val_ds = PartDataset(data_path, 1, batch_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=threads)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=threads)
    return train_loader, val_loader


def class_weights(label, num_classes, device):
    """Inverse-frequency class weights to counter foreground/background imbalance."""
    target = label[..., 0].cpu().numpy().flatten().astype("i8")
    target = np.clip(target, 0, num_classes - 1)          # guard vs. stray labels
    counts = np.bincount(target, minlength=num_classes)[:num_classes].astype("f4")
    freq = counts / max(counts.sum(), 1)
    # weight = 1 / (frequency * num_classes); guard against zeros
    w = 1.0 / np.maximum(freq * num_classes, 1e-6)
    w = w / w.sum() * num_classes
    return torch.from_numpy(w).to(device)


def train_one_epoch(model, loader, optimizer, device, cc):
    """Run a single pass over `loader`; returns averaged metrics."""
    model.train()
    tot = {"loss": 0.0, "cls": 0.0, "part": 0.0, "acc": 0.0, "dice": 0.0}
    n = 0

    for i, (image, label, cur_xyz, scale, cur_xyz_o) in enumerate(loader):
        # image: (1,1,nx,ny,nz) occupancy volume
        # label: (1, N, 3) -> [cls, part, roi_index]
        image = image.to(device).float()
        label = label.to(device)

        w = class_weights(label, model.K, device)

        optimizer.zero_grad()
        output, fts = model(image)

        # gather predictions/features at the occupied-voxel indices (roi)
        roi = label[0, ..., -1]
        output = output.contiguous().index_select(0, roi)
        fts = fts.contiguous().index_select(0, roi)

        cls_loss = F.nll_loss(output, label[0, ..., 0], weight=w)
        part_loss = disc_loss(fts, label[0, ..., 1], cc)
        loss = cls_loss + part_loss
        loss.backward()

        if conf.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip)
        optimizer.step()

        with torch.no_grad():
            pred = output.argmax(1)
            true = label[0, ..., 0]
            acc = pred.eq(true).float().mean().item() * 100
            inter = (pred * true).sum().float()
            dice = (2 * inter / (pred.sum() + true.sum()).clamp(min=1)).item() * 100

        tot["loss"] += loss.item()
        tot["cls"] += cls_loss.item()
        tot["part"] += part_loss.item()
        tot["acc"] += acc
        tot["dice"] += dice
        n += 1

        if i % 50 == 0:
            logging.info("  batch %d | loss %.5f | cls %.5f | part %.5f | acc %.2f%%",
                         i, loss.item(), cls_loss.item(), part_loss.item(), acc)

    return {k: v / max(n, 1) for k, v in tot.items()}


def main():
    parser = argparse.ArgumentParser(description="Train VCNN")
    parser.add_argument("--data", default=None, help="path to training .h5 file")
    parser.add_argument("--val", default=None, help="path to validation .h5 file")
    parser.add_argument("--model", default="vcnn", help="model name prefix")
    parser.add_argument("--resume", type=int, default=0,
                        help="resume from checkpoint at this epoch (0 = scratch)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    # ---- resolve config from CLI overrides ---- #
    if args.epochs is not None:
        conf.epochs = args.epochs
    if args.lr is not None:
        conf.learning_rate = args.lr
    if args.batch_size is not None:
        conf.batch_size = args.batch_size

    conf.ensure_dirs()
    logging.info("platform=%s device=%s", conf.platform, conf.device)
    logging.info("\n%s", conf.summary())

    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    if conf.cuda:
        torch.cuda.manual_seed(conf.seed)

    # ---- data ---- #
    train_path = args.data or os.path.join(
        conf.data_dir, "x_y1_y2_idx_33_withoutnoise.h5")
    val_path = args.val or train_path
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "training data not found: %s\nPass --data <path> or set VCNN_DATA_DIR."
            % train_path)

    train_loader, val_loader = build_loaders(train_path, conf.batch_size, conf.threads)
    logging.info("train file: %s", train_path)
    logging.info("val   file: %s", val_path)

    # ---- model / optimizer ---- #
    model = VCNN(conf.num_classes, conf.criterion).to(conf.device)
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)

    start_epoch = 0
    if args.resume > 0:
        ckpt_path = os.path.join(
            conf.ckpt_dir, "%s_%03d.pth" % (args.model, args.resume))
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=conf.device))
            start_epoch = args.resume
            logging.info("resumed from %s (epoch %d)", ckpt_path, start_epoch)
        else:
            logging.warning("resume checkpoint not found: %s", ckpt_path)
    if start_epoch == 0:
        model.apply(weights_init)

    n_params = sum(p.numel() for p in model.parameters())
    logging.info("number of parameters: %d", n_params)

    cc = conf.disc_loss_conf
    log_path = os.path.join(conf.log_dir, "loss.txt")

    # ---- training loop ---- #
    for epoch in range(start_epoch, conf.epochs):
        stats = train_one_epoch(model, train_loader, optimizer, conf.device, cc)
        logging.info(
            "===> epoch %d/%d | loss %.5f | cls %.5f | part %.5f | acc %.2f%% | dice %.2f%%",
            epoch + 1, conf.epochs, stats["loss"], stats["cls"], stats["part"],
            stats["acc"], stats["dice"])

        with open(log_path, "a") as f:
            f.write("epoch:%d loss:%.6f cls:%.6f part:%.6f acc:%.4f dice:%.4f lr:%s\n"
                    % (epoch + 1, stats["loss"], stats["cls"], stats["part"],
                       stats["acc"], stats["dice"], conf.learning_rate))

        # checkpoint (so we can resume after a Kaggle/Colab session cut)
        if (epoch + 1) % conf.save_every == 0:
            ckpt = os.path.join(conf.ckpt_dir, "%s_%03d.pth" % (args.model, epoch + 1))
            torch.save(model.state_dict(), ckpt)
            logging.info("saved checkpoint: %s", ckpt)

        # learning-rate decay for the first 100 epochs (as in the paper)
        if (epoch + 1) % 20 == 0 and (epoch + 1) <= 100:
            conf.learning_rate = conf.learning_rate / np.sqrt(epoch + 1)
            for g in optimizer.param_groups:
                g["lr"] = conf.learning_rate

    # final checkpoint
    final = os.path.join(conf.ckpt_dir, "%s_final.pth" % args.model)
    torch.save(model.state_dict(), final)
    logging.info("training complete. final model: %s", final)


if __name__ == "__main__":
    main()


