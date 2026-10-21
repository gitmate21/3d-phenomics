"""
Central configuration for the Eff-3DPSeg port (spconv Res16UNet34C).

Paths are resolved in this order (highest priority first):
    1. CLI argument (handled in scripts/train.py and scripts/infer.py)
    2. Environment variable (EFF_DATA_DIR, EFF_CKPT_DIR, EFF_LOG_DIR)
    3. Platform-aware default (Kaggle / Colab / local)

Only light imports happen here so that the data scripts work without torch.
"""

from __future__ import annotations

import os


def detect_platform() -> str:
    """Return 'kaggle', 'colab', or 'local' based on the runtime environment."""
    if os.path.isdir("/kaggle"):
        return "kaggle"
    if os.path.isdir("/content"):
        return "colab"
    return "local"


def _default_dirs(platform: str, repo_root: str) -> dict:
    data_dir = os.path.join(repo_root, "data")
    if platform == "kaggle":
        # /kaggle/working persists as notebook output.
        base = "/kaggle/working"
        return {
            "data": data_dir,
            "ckpt": os.path.join(base, "checkpoints"),
            "log": os.path.join(base, "logs"),
        }
    if platform == "colab":
        drive = "/content/drive/MyDrive"
        base = os.path.join(drive, "eff3dpseg") if os.path.isdir(drive) else "/content"
        return {
            "data": data_dir,
            "ckpt": os.path.join(base, "checkpoints"),
            "log": os.path.join(base, "logs"),
        }
    return {
        "data": data_dir,
        "ckpt": os.path.join(repo_root, "checkpoints"),
        "log": os.path.join(repo_root, "logs"),
    }


class Config:
    """Runtime configuration + hyper-parameters (defaults follow pheno4d.sh)."""

    def __init__(self) -> None:
        self.platform = detect_platform()
        self.repo_root = os.path.dirname(os.path.abspath(__file__))

        d = _default_dirs(self.platform, self.repo_root)
        self.data_dir = os.environ.get("EFF_DATA_DIR", d["data"])
        self.ckpt_dir = os.environ.get("EFF_CKPT_DIR", d["ckpt"])
        self.log_dir = os.environ.get("EFF_LOG_DIR", d["log"])

        # ---- labels -------------------------------------------------------- #
        # Raw Pheno4D ids (after prepare_pheno4d.py): 0 soil, 1 stem, 2 leaf.
        # Soil is ignored in the loss; the network predicts stem (0) / leaf (1).
        self.class_names = ("stem", "leaf")
        self.num_classes = 2
        self.ignore_label = 255

        # ---- model --------------------------------------------------------- #
        self.model = "Res16UNet34C"
        self.feature_mode = "ones"      # 'ones' (geometry only, as original) or 'xyz'
        self.conv1_kernel_size = 5      # first conv; original config default
        self.bn_momentum = 0.02

        # ---- voxelization / augmentation ---------------------------------- #
        # voxel_size is in the units of the raw point cloud. The original used
        # 0.2 ("2cm" dataset); check your data units and override with --voxel-size.
        self.voxel_size = 0.2
        self.scale_aug = (0.9, 1.1)
        # (x, y, z) rotation bounds in radians; z is the upright axis.
        self.rotation_aug = (
            (-3.141592653589793 / 64, 3.141592653589793 / 64),
            (-3.141592653589793 / 64, 3.141592653589793 / 64),
            (-3.141592653589793, 3.141592653589793),
        )
        # (granularity, magnitude) pairs in voxel units; one pair is sampled per sample.
        self.elastic_params = ((10.0, 2.0), (40.0, 8.0))
        self.point_dropout = 0.2

        # ---- supervision (Eff-3DPSeg paper, Sec. 2.2.4) -------------------- #
        # labeled_points = 0 -> every point labeled (full supervision). k > 0 -> keep k randomly chosen
        # labeled points per training cloud (after random down-sampling by weak_downsample); every other
        # point is unlabeled (ignored in the loss). Validation/test clouds are always fully labeled.
        self.labeled_points = 0
        self.weak_downsample = 0.2
        self.label_seed = 0
        # The paper deletes the soil points. True removes them from the input as well as from the loss;
        # False keeps them as input and only ignores them in the loss (the behaviour of earlier runs).
        self.drop_soil = True

        # ---- optimisation (pheno4d.sh) ------------------------------------ #
        self.lr = 0.1
        self.momentum = 0.9
        self.dampening = 0.1
        self.weight_decay = 1e-4
        self.poly_power = 0.9
        self.max_iter = 2000
        self.train_batch_size = 2
        self.val_batch_size = 1
        self.optim_step = 1             # gradient accumulation steps
        self.validate_step = 50
        self.save_every = 100           # iterations between 'latest' checkpoints
        self.workers = 2
        self.seed = 42
        self.grad_clip = None

    # ------------------------------------------------------------------ #
    def ensure_dirs(self) -> None:
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def summary(self) -> str:
        lines = [
            "============== Eff-3DPSeg configuration ==============",
            f"platform      : {self.platform}",
            f"repo_root     : {self.repo_root}",
            f"data_dir      : {self.data_dir}",
            f"ckpt_dir      : {self.ckpt_dir}",
            f"log_dir       : {self.log_dir}",
            f"model         : {self.model}",
            f"voxel_size    : {self.voxel_size}",
            f"feature_mode  : {self.feature_mode}",
            f"lr / max_iter : {self.lr} / {self.max_iter}",
            f"batch (train) : {self.train_batch_size} x accum {self.optim_step}",
            "======================================================",
        ]
        return "\n".join(lines)


conf = Config()
