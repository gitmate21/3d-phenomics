"""
Train Res16UNet34C (spconv or pure-PyTorch sparse conv) on Pheno4D stem/leaf segmentation.

    python scripts/train.py --data /kaggle/input/pheno4d-npz --run-name demo --max-iter 2000

Single GPU, mixed precision, gradient accumulation, polynomial LR, validation mIoU
every --validate-step iterations, 'latest' and 'best' checkpoints, full resume
(model + optimizer + scaler + iteration) with --resume auto|PATH.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from config import conf as default_conf  # noqa: E402
from datasets import Pheno4DDataset, collate  # noqa: E402
from models import build_model, to_sparse  # noqa: E402
from utils import fast_hist, per_class_iu, set_seed  # noqa: E402


def parse_args(argv=None):
    c = default_conf
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=c.data_dir, help="root with <Plant>/*.npz and splits/")
    p.add_argument("--ckpt-dir", default=c.ckpt_dir)
    p.add_argument("--log-dir", default=c.log_dir)
    p.add_argument("--run-name", default="eff3dpseg")
    p.add_argument("--train-phase", default="train", choices=["train", "trainval"])
    p.add_argument("--val-phase", default="val", choices=["val", "test", "none"])
    p.add_argument("--max-iter", type=int, default=c.max_iter)
    p.add_argument("--lr", type=float, default=c.lr)
    p.add_argument("--batch-size", type=int, default=c.train_batch_size)
    p.add_argument("--optim-step", type=int, default=c.optim_step, help="gradient accumulation steps")
    p.add_argument("--validate-step", type=int, default=c.validate_step)
    p.add_argument("--save-every", type=int, default=c.save_every)
    p.add_argument("--workers", type=int, default=c.workers)
    p.add_argument("--voxel-size", type=float, default=c.voxel_size)
    p.add_argument("--feature-mode", default=c.feature_mode, choices=["ones", "xyz"])
    p.add_argument("--seed", type=int, default=c.seed)
    p.add_argument("--weights", default=None, help="initialise model from a checkpoint (not optimizer)")
    p.add_argument("--resume", default=None, help="'auto' (latest in ckpt-dir) or a checkpoint path")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--time-limit-hours", type=float, default=None,
                   help="save and stop after this wall time (Kaggle sessions cap at ~12h)")
    return p.parse_args(argv)


def make_conf(args):
    conf = copy.deepcopy(default_conf)
    conf.voxel_size = args.voxel_size
    conf.feature_mode = args.feature_mode
    return conf


def poly_lr(base, it, max_iter, power):
    return base * (1 - it / (max_iter + 1)) ** power


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=True)


@torch.no_grad()
def evaluate(model, loader, device, n_cls, ignore, amp):
    model.eval()
    hist = np.zeros((n_cls, n_cls), dtype=np.int64)
    total, count = 0.0, 0
    for batch in loader:
        labels = batch["labels"].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(to_sparse(batch, device))
        logits = out.features.float()
        if (labels != ignore).any():
            total += F.cross_entropy(logits, labels, ignore_index=ignore).item()
            count += 1
        hist += fast_hist(logits.argmax(1).cpu().numpy(), labels.cpu().numpy(), n_cls)
    model.train()
    ious = per_class_iu(hist)
    return total / max(count, 1), float(np.nanmean(ious)), ious


def main(argv=None):
    args = parse_args(argv)
    conf = make_conf(args)
    if not torch.cuda.is_available():
        sys.exit("training needs a CUDA GPU (Kaggle: Settings -> Accelerator -> GPU).")
    device = torch.device("cuda")
    amp = not args.no_amp
    set_seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_ds = Pheno4DDataset(args.data, args.train_phase, conf, train=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=len(train_ds) >= args.batch_size)
    val_loader = None
    if args.val_phase != "none":
        val_ds = Pheno4DDataset(args.data, args.val_phase, conf, train=False)
        val_loader = DataLoader(val_ds, batch_size=conf.val_batch_size, shuffle=False, collate_fn=collate,
                                num_workers=args.workers)
    print(f"train clouds: {len(train_ds)}  val clouds: {len(val_loader.dataset) if val_loader else 0}")

    in_ch = 1 if conf.feature_mode == "ones" else 3
    model = build_model(conf.model, in_ch, conf.num_classes, conf.conv1_kernel_size, conf.bn_momentum).to(device)
    print(f"model {conf.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=conf.momentum,
                                dampening=conf.dampening, weight_decay=conf.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    it, best = 0, -1.0
    latest = os.path.join(args.ckpt_dir, f"{args.run_name}_latest.pth")
    best_path = os.path.join(args.ckpt_dir, f"{args.run_name}_best.pth")
    resume = latest if args.resume == "auto" else args.resume
    if resume and os.path.isfile(resume):
        ck = load_ckpt(resume)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        it, best = ck["iteration"], ck["best_miou"]
        print(f"resumed from {resume} at iteration {it} (best mIoU {best:.4f})")
    elif args.resume not in (None, "auto"):
        sys.exit(f"--resume checkpoint not found: {args.resume}")
    elif args.weights:
        res = model.load_state_dict(load_ckpt(args.weights)["model"], strict=False)
        print(f"loaded weights from {args.weights}; missing={len(res.missing_keys)} "
              f"unexpected={len(res.unexpected_keys)}")

    def save(path):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "iteration": it, "best_miou": best,
                    "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}},
                   path)

    start_it = it
    log = open(os.path.join(args.log_dir, f"{args.run_name}_log.txt"), "a")
    data_iter = infinite(train_loader)
    model.train()
    t0, run_loss, run_n = time.time(), 0.0, 0

    while it < args.max_iter:
        lr = poly_lr(args.lr, it, args.max_iter, conf.poly_power)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        did_backward = False
        for _ in range(args.optim_step):
            batch = next(data_iter)
            labels = batch["labels"].to(device)
            if not (labels != conf.ignore_label).any():
                continue
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                out = model(to_sparse(batch, device))
            loss = F.cross_entropy(out.features.float(), labels, ignore_index=conf.ignore_label)
            scaler.scale(loss / args.optim_step).backward()
            run_loss += loss.item()
            run_n += 1
            did_backward = True
        if did_backward:
            scaler.step(optimizer)
            scaler.update()
        it += 1

        if it % 10 == 0 and run_n:
            print(f"iter {it}/{args.max_iter}  lr {lr:.5f}  loss {run_loss / run_n:.4f}  "
                  f"{(time.time() - t0) / (it - start_it):.2f}s/it")
            log.write(f"{it},train_loss,{run_loss / run_n:.6f}\n")
            run_loss, run_n = 0.0, 0

        timed_out = (args.time_limit_hours is not None
                     and (time.time() - t0) > args.time_limit_hours * 3600)

        if val_loader and (it % args.validate_step == 0 or it == args.max_iter):
            vloss, miou, ious = evaluate(model, val_loader, device, conf.num_classes, conf.ignore_label, amp)
            print(f"[val] iter {it}  loss {vloss:.4f}  mIoU {miou:.4f}  "
                  + "  ".join(f"{n} {i:.4f}" for n, i in zip(conf.class_names, ious)))
            log.write(f"{it},val_miou,{miou:.6f}\n")
            if miou > best:
                best = miou
                save(best_path)
                print(f"  new best -> {best_path}")
        log.flush()

        if it % args.save_every == 0 or it == args.max_iter or timed_out:
            save(latest)
        if timed_out:
            print("time limit reached; checkpoint saved, stopping.")
            break

    if not val_loader:
        save(best_path)
    log.close()
    print("done.")


if __name__ == "__main__":
    main()
