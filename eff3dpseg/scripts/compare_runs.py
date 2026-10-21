"""
Evaluate several trained runs and print them side by side, next to the matching Eff-3DPSeg paper rows.

    python scripts/compare_runs.py --data DATA --ckpt-dir CKPT --runs tomato_full tomato_w50 tomato_w100 tomato_w200

Each run's voxel size, feature mode and soil handling are read from its checkpoint. Numbers are computed on all
raw points (soil excluded), pooled over the clouds of each phase. The paper rows are Table 3 (tomato / Pheno4D):
"Baseline" = no self-supervised pretraining (the closest match to runs made with this code), "Eff-3DPSeg" = pretrained.
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
from utils import fast_hist, format_metrics, prf_iou  # noqa: E402

# Paper Table 3, tomato (Pheno4D). Each metric is (stem, leaf, mean) in %.
# key: labeled points per cloud (0 = full supervision) -> {"Baseline": ..., "Eff-3DPSeg": ...}
PAPER = {
    50: {"Baseline": {"Precision": (76.6, 96.3, 86.4), "Recall": (75.3, 96.5, 85.9), "F1": (75.9, 96.4, 86.2), "IoU": (61.2, 93.1, 77.1)},
         "Eff-3DPSeg": {"Precision": (86.6, 97.1, 91.8), "Recall": (81.4, 98.0, 89.7), "F1": (83.9, 97.5, 90.7), "IoU": (72.3, 95.2, 83.7)}},
    100: {"Baseline": {"Precision": (66.4, 95.8, 81.1), "Recall": (69.9, 95.1, 82.5), "F1": (68.1, 95.4, 81.7), "IoU": (51.6, 91.2, 71.4)},
          "Eff-3DPSeg": {"Precision": (86.8, 98.1, 92.4), "Recall": (86.9, 98.0, 92.5), "F1": (86.9, 98.1, 92.5), "IoU": (76.8, 96.2, 86.5)}},
    200: {"Baseline": {"Precision": (87.2, 98.3, 92.8), "Recall": (88.5, 98.1, 93.3), "F1": (87.8, 98.2, 93.0), "IoU": (78.3, 96.5, 87.4)},
          "Eff-3DPSeg": {"Precision": (91.5, 98.0, 94.7), "Recall": (87.0, 98.7, 92.9), "F1": (89.2, 98.4, 93.8), "IoU": (80.5, 96.8, 88.6)}},
    0: {"Eff-3DPSeg": {"Precision": (94.7, 99.4, 97.1), "Recall": (95.9, 99.2, 97.5), "F1": (95.3, 99.3, 97.3), "IoU": (91.0, 98.6, 94.8)}},
}


def paper_lines(k: int) -> list:
    rows = PAPER.get(k)
    if not rows:
        return [f"  (no paper row for {k} labeled points)"]
    out = []
    for name, m in rows.items():
        out.append(f"  paper {name:<11}" + "   ".join(
            f"{metric} {m[metric][0]:.1f}/{m[metric][1]:.1f}/{m[metric][2]:.1f}" for metric in ("Precision", "Recall", "F1", "IoU")))
    return out


def evaluate(ckpt_path, data, phases, device, workers, amp):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    saved = ck.get("args", {})
    conf = copy.deepcopy(default_conf)
    conf.voxel_size = saved.get("voxel_size", conf.voxel_size)
    conf.feature_mode = saved.get("feature_mode", conf.feature_mode)
    conf.drop_soil = saved.get("drop_soil", False)            # checkpoints from before the option kept soil
    in_ch = 1 if conf.feature_mode == "ones" else 3
    model = build_model(conf.model, in_ch, conf.num_classes, conf.conv1_kernel_size, conf.bn_momentum).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    n = conf.num_classes
    results = {}
    for phase in phases:
        ds = Pheno4DDataset(data, phase, conf, train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=workers)
        hist = np.zeros((n, n), dtype=np.int64)
        with torch.no_grad():
            for b in loader:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    out = model(to_sparse(b, device))
                pred = out.features.argmax(1).cpu().numpy()[b["inverse"][0]]
                hist += fast_hist(pred, b["point_labels"][0], n)
        results[phase] = (hist, len(ds))
    return results, saved, ck, conf


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=default_conf.data_dir)
    p.add_argument("--ckpt-dir", default=default_conf.ckpt_dir)
    p.add_argument("--runs", nargs="+", required=True, help="run names, e.g. tomato_full tomato_w50")
    p.add_argument("--which", default="best", choices=["best", "latest"])
    p.add_argument("--phases", nargs="+", default=["val", "test"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args(argv)

    cuda = torch.cuda.is_available()
    if not cuda and not args.allow_cpu:
        sys.exit("needs a CUDA GPU (use --allow-cpu only for tests).")
    device = torch.device("cuda" if cuda else "cpu")

    summary = []
    for run in args.runs:
        path = os.path.join(args.ckpt_dir, f"{run}_{args.which}.pth")
        if not os.path.isfile(path):
            print(f"\n!! {path} not found, skipping {run}")
            continue
        results, saved, ck, conf = evaluate(path, args.data, args.phases, device, args.workers, cuda)
        k = int(saved.get("labeled_points", 0))
        print(f"\n################ {run} ({args.which}, iteration {ck.get('iteration')}) ################")
        print(f"trained with: labeled_points={saved.get('labeled_points', 'all (legacy)')}  "
              f"drop_soil={conf.drop_soil}  voxel_size={conf.voxel_size}  max_iter={saved.get('max_iter')}  "
              f"batch_size={saved.get('batch_size')}  data={saved.get('data')}")
        for phase, (hist, n_clouds) in results.items():
            t = prf_iou(hist)
            print(f"\n-- {phase} ({n_clouds} clouds, pooled over all points)")
            print(format_metrics(t, conf.class_names))
            summary.append((run, phase, k, t))
        print("\n".join(paper_lines(k)))

    if summary:
        print("\n\n==================== summary (IoU and F1 in %, stem / leaf / mean) ====================")
        print(f"{'run':<18}{'phase':<7}{'labeled':>8}   {'IoU stem':>9}{'leaf':>7}{'mean':>7}   {'F1 mean':>8}{'Prec mean':>10}{'Rec mean':>9}")
        for run, phase, k, t in summary:
            print(f"{run:<18}{phase:<7}{('all' if k == 0 else k):>8}   {t[0, 3]:9.1f}{t[1, 3]:7.1f}{np.nanmean(t[:, 3]):7.1f}"
                  f"   {np.nanmean(t[:, 2]):8.1f}{np.nanmean(t[:, 0]):10.1f}{np.nanmean(t[:, 1]):9.1f}")


if __name__ == "__main__":
    main()
