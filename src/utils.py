"""
Utility helpers shared across the boundary detection pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import yaml


@dataclass
class Config:
    data_dir: Path
    features_dir: Path
    results_dir: Path
    fps: float
    resize: Sequence[int]
    window: int
    stride: int
    batch_size: int
    nms_window: int
    feature_dim: int
    pca_rank: int
    tolerance_sec: float
    device: torch.device
    seed: int
    alpha: Optional[float]
    gt_path: Path

    def __getitem__(self, key: str):
        """Dict-like access for convenience."""
        return getattr(self, key)

    def keys(self) -> Iterable[str]:
        return self.__dataclass_fields__.keys()

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.keys()}


def load_config(config_path: str | Path) -> Config:
    """
    Parse a YAML configuration file into a Config dataclass.
    Adds derived paths and converts primitive types where needed.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)

    config_dir = path.parent
    project_root = config_dir.parent if config_dir.name == "configs" else config_dir
    alpha_value = raw_cfg.get("alpha", None)
    gt_name = raw_cfg.get("gt_file", "data/breakfast_gt.txt")

    def resolve_path(value: str | Path, prefer_root: bool = False) -> Path:
        path_value = Path(value)
        if path_value.is_absolute():
            return path_value
        root_candidate = (project_root / path_value).resolve()
        config_candidate = (config_dir / path_value).resolve()
        cwd_candidate = (Path.cwd() / path_value).resolve()
        if prefer_root:
            return root_candidate
        for candidate in (cwd_candidate, root_candidate, config_candidate):
            if candidate.exists():
                return candidate
        return root_candidate

    data_dir = resolve_path(raw_cfg["data_dir"])
    features_dir = resolve_path(raw_cfg["features_dir"], prefer_root=True)
    results_dir = resolve_path(raw_cfg["results_dir"], prefer_root=True)
    gt_path = resolve_path(gt_name)

    cfg = Config(
        data_dir=data_dir,
        features_dir=features_dir,
        results_dir=results_dir,
        fps=float(raw_cfg["fps"]),
        resize=tuple(raw_cfg.get("resize", [256, 256])),
        window=int(raw_cfg["window"]),
        stride=int(raw_cfg["stride"]),
        batch_size=int(raw_cfg.get("batch_size", 32)),
        nms_window=int(raw_cfg["nms_window"]),
        feature_dim=int(raw_cfg["feature_dim"]),
        pca_rank=int(raw_cfg["pca_rank"]),
        tolerance_sec=float(raw_cfg["tolerance_sec"]),
        device=torch.device(raw_cfg.get("device", "cpu")),
        seed=int(raw_cfg.get("seed", 0)),
        alpha=float(alpha_value) if alpha_value is not None else None,
        gt_path=gt_path,
    )

    cfg.features_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def set_random_seed(seed: int) -> None:
    """Seed random, numpy and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_video_ids(data_dir: Path) -> List[str]:
    """Return sorted video identifiers (directory names) under data_dir."""
    return sorted([p.name for p in Path(data_dir).iterdir() if p.is_dir()])


def list_frame_paths(video_dir: Path) -> List[Path]:
    """Collect sorted frame paths for a single video directory."""
    frames = [p for p in video_dir.glob("*.jpg")]
    return sorted(frames)


def load_ground_truth(path: Path) -> Dict[str, List[int]]:
    """
    Load ground-truth boundaries.
    Expected format per line: <video_id> <frame_idx> [<frame_idx> ...]
    Returns dictionary mapping video_id -> sorted list of frame indices.
    """
    gt: Dict[str, List[int]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            vid, *frames = parts
            gt[vid] = sorted(int(f) for f in frames)
    return gt


def frame_to_seconds(frame_idx: int, fps: float) -> float:
    """Convert a frame index into seconds."""
    return frame_idx / fps


def seconds_to_frame(second: float, fps: float) -> int:
    """Convert seconds back to the nearest frame index."""
    return int(round(second * fps))


__all__ = [
    "Config",
    "load_config",
    "set_random_seed",
    "list_video_ids",
    "list_frame_paths",
    "load_ground_truth",
    "frame_to_seconds",
    "seconds_to_frame",
]
