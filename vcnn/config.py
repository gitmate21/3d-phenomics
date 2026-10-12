"""
Central configuration for the VCNN (voxel-based CNN).

All paths and hyper-parameters live here as a single source of truth. Paths are
resolved in this order (highest priority first):
    1. CLI argument (handled in train.py)
    2. Environment variable (VCNN_DATA_DIR, VCNN_CKPT_DIR, VCNN_LOG_DIR)
    3. Platform-aware default (Kaggle / Colab / local)
"""

from __future__ import annotations

import os

import torch


# --------------------------------------------------------------------------- #
# Platform detection
# --------------------------------------------------------------------------- #
def detect_platform() -> str:
    """Return 'kaggle', 'colab', or 'local' based on the runtime environment."""
    if os.path.isdir("/kaggle"):
        return "kaggle"
    if os.path.isdir("/content"):
        return "colab"
    return "local"


def _default_dirs(platform: str, repo_root: str) -> dict:
    """Return sensible default directories for the detected platform."""
    # The demo dataset ships inside the repo at <repo_root>/data, so that is the
    # default data location on every platform. Users override it (for their own
    # large datasets) with --data or the VCNN_DATA_DIR env var.
    data_dir = os.path.join(repo_root, "data")

    if platform == "kaggle":
        # /kaggle/working persists as notebook output; good for checkpoints/logs.
        base = "/kaggle/working"
        return {
            "data": data_dir,
            "ckpt": os.path.join(base, "checkpoints"),
            "log": os.path.join(base, "logs"),
        }
    if platform == "colab":
        # Prefer Google Drive for persistence across disconnects, else /content.
        drive = "/content/drive/MyDrive"
        base = os.path.join(drive, "vcnn") if os.path.isdir(drive) else "/content"
        return {
            "data": data_dir,
            "ckpt": os.path.join(base, "checkpoints"),
            "log": os.path.join(base, "logs"),
        }
    # local
    return {
        "data": data_dir,
        "ckpt": os.path.join(repo_root, "checkpoints"),
        "log": os.path.join(repo_root, "logs"),
    }


class Config:
    """Runtime configuration + hyper-parameters for training the VCNN."""

    def __init__(self) -> None:
        self.platform = detect_platform()
        self.repo_root = os.path.dirname(os.path.abspath(__file__))

        # ---- device -------------------------------------------------------- #
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        # ---- directories (overridable via env vars) ----------------------- #
        d = _default_dirs(self.platform, self.repo_root)
        self.data_dir = os.environ.get("VCNN_DATA_DIR", d["data"])
        self.ckpt_dir = os.environ.get("VCNN_CKPT_DIR", d["ckpt"])
        self.log_dir = os.environ.get("VCNN_LOG_DIR", d["log"])

        # ---- model / training hyper-parameters ---------------------------- #
        self.num_classes = 2            # 0 = background/noise, 1 = plant part
        self.criterion = "nll"          # 'nll' or 'dice'
        self.learning_rate = 1e-5
        self.batch_size = 1
        self.epochs = 1000
        self.threads = 0                # DataLoader workers (0 = main process)
        self.seed = 714
        self.from_scratch = True
        self.save_every = 2             # save a checkpoint every N epochs
        self.prefix = "vcnn"

        # Discriminative (instance-embedding) loss weights.
        #   c0   : intra-cluster variance threshold (delta_v)
        #   c1   : inter-cluster distance threshold (delta_d)
        #   abcd : weights for [variance, distance, center-norm, center-regularizer]
        self.disc_loss_conf = {
            "c0": 1,
            "c1": 3,
            "abcd": [1, 1, 0.01, 0.01],
        }

        # Gradient clipping for training stability (set to None to disable).
        self.grad_clip = 1.0

        self._pid = os.getpid()

    # ------------------------------------------------------------------ #
    def ensure_dirs(self) -> None:
        """Create checkpoint/log directories if they do not exist."""
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def summary(self) -> str:
        lines = [
            "================ VCNN configuration ================",
            f"platform      : {self.platform}",
            f"device        : {self.device} (cuda={self.cuda})",
            f"repo_root     : {self.repo_root}",
            f"data_dir      : {self.data_dir}",
            f"ckpt_dir      : {self.ckpt_dir}",
            f"log_dir       : {self.log_dir}",
            f"num_classes   : {self.num_classes}",
            f"criterion     : {self.criterion}",
            f"learning_rate : {self.learning_rate}",
            f"batch_size    : {self.batch_size}",
            f"epochs        : {self.epochs}",
            "====================================================",
        ]
        return "\n".join(lines)


# A module-level singleton for convenience.
conf = Config()
