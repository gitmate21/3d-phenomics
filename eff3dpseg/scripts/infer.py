"""
Full-resolution inference + evaluation.

    python scripts/infer.py --data DATA --ckpt checkpoints/demo_best.pth --phase test --out-dir preds

Each plant is voxelized without augmentation, predicted, and the voxel labels are
copied back to every raw point through the voxelization's inverse map (exact, no
KD-tree needed). IoU is computed on raw points; soil / unlabeled points are skipped.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from config import conf as default_conf  # noqa: E402
from datasets import Pheno4DDataset, collate  # noqa: E402
from models import build_model, to_sparse  # noqa: E402
from utils import fast_hist, format_metrics, per_class_iu, prf_iou, save_ply  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=default_conf.data_dir)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--phase", default="test", choices=["train", "val", "test", "trainval"])
    p.add_argument("--out-dir", default=None, help="write per-plant .npz predictions here")
    p.add_argument("--save-ply", action="store_true", help="also write coloured .ply files")
    p.add_argument("--voxel-size", type=float, default=default_conf.voxel_size)
    p.add_argument("--feature-mode", default=default_conf.feature_mode, choices=["ones", "xyz"])
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        sys.exit("inference needs a CUDA GPU.")
    device = torch.device("cuda")
    conf = copy.deepcopy(default_conf)
    conf.voxel_size, conf.feature_mode = args.voxel_size, args.feature_mode

    in_ch = 1 if conf.feature_mode == "ones" else 3
    model = build_model(conf.model, in_ch, conf.num_classes, conf.conv1_kernel_size, conf.bn_momentum).to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = Pheno4DDataset(args.data, args.phase, conf, train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=2)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    n = conf.num_classes
    hist = np.zeros((n, n), dtype=np.int64)
    per_cloud = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=not args.no_amp):
                out = model(to_sparse(batch, device))
            vox_pred = out.features.argmax(1).cpu().numpy()
            point_pred = vox_pred[batch["inverse"][0]]
            point_label = batch["point_labels"][0]
            h = fast_hist(point_pred, point_label, n)
            hist += h
            per_cloud.append(h)
            name = batch["name"][0]
            iou = per_class_iu(h)
            print(f"{name}: mIoU {np.nanmean(iou):.4f}  " +
                  "  ".join(f"{c} {i:.4f}" for c, i in zip(conf.class_names, iou)))
            if args.out_dir:
                stem = name.replace("/", "_").rsplit(".", 1)[0]
                np.savez_compressed(os.path.join(args.out_dir, stem + "_pred.npz"),
                                    xyz=batch["xyz"][0], pred=point_pred.astype(np.uint8), label=point_label.astype(np.uint8))
                if args.save_ply:
                    save_ply(os.path.join(args.out_dir, stem + "_pred.ply"), batch["xyz"][0], point_pred)

    ious = per_class_iu(hist)
    print("\n==== overall (raw points) ====")
    print("mIoU:", f"{np.nanmean(ious) * 100:.2f}")
    for c, i in zip(conf.class_names, ious):
        print(f"  {c}: {i * 100:.2f}")
    print("confusion matrix (rows = ground truth):\n", hist)

    # Paper-style metrics (Eff-3DPSeg Table 3): per class and the mean over the classes, in %.
    print("\n==== precision / recall / F1 / IoU, pooled over all points ====")
    print(format_metrics(prf_iou(hist), conf.class_names))
    if per_cloud:
        mean_cloud = np.nanmean(np.stack([prf_iou(h) for h in per_cloud]), axis=0)
        print(f"\n==== same metrics, averaged over the {len(per_cloud)} clouds ====")
        print(format_metrics(mean_cloud, conf.class_names))


if __name__ == "__main__":
    main()
