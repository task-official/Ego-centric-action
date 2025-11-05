"""
Feature extraction script.
Loads frames for each video, runs a pretrained ResNet18 (ImageNet) backbone,
and saves per-frame features to .npy files under the configured features_dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

import utils


class FrameDataset(Dataset):
    """PyTorch dataset for loading video frames from disk."""

    def __init__(self, frame_paths: List[Path], transform: transforms.Compose):
        self.frame_paths = frame_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        return self.transform(img)


def build_model(device: torch.device, resize: Tuple[int, int]) -> Tuple[torch.nn.Module, transforms.Compose]:
    """Instantiate a pretrained ResNet18 truncated before the classifier."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    meta = getattr(weights, "meta", {}) or {}
    mean = meta.get("mean", (0.485, 0.456, 0.406))
    std = meta.get("std", (0.229, 0.224, 0.225))
    preprocess = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return model, preprocess


@torch.inference_mode()
def extract_video_features(
    video_id: str,
    cfg: utils.Config,
    model: torch.nn.Module,
    preprocess: transforms.Compose,
) -> np.ndarray:
    """Run the feature extractor for a single video and return stacked features."""
    video_dir = cfg.data_dir / video_id
    frame_paths = utils.list_frame_paths(video_dir)
    if not frame_paths:
        return np.empty((0, cfg.feature_dim), dtype=np.float32)

    dataset = FrameDataset(frame_paths, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cfg.device.type == "cuda",
    )

    feats = []
    for batch in tqdm(loader, desc=f"Extract {video_id}", leave=False):
        batch = batch.to(cfg.device, non_blocking=True)
        outputs = model(batch)
        feats.append(outputs.detach().cpu().numpy())

    features = np.concatenate(feats, axis=0)
    return features.astype(np.float32, copy=False)


def save_features(output_path: Path, features: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)


def run(cfg: utils.Config) -> None:
    utils.set_random_seed(cfg.seed)
    resize = tuple(cfg.resize)
    model, preprocess = build_model(cfg.device, resize)

    video_ids = utils.list_video_ids(cfg.data_dir)
    for vid in video_ids:
        output_path = cfg.features_dir / f"{vid}.npy"
        if output_path.exists():
            continue
        start = time.time()
        features = extract_video_features(vid, cfg, model, preprocess)
        save_features(output_path, features)
        elapsed = time.time() - start
        frame_count = features.shape[0]
        if frame_count > 0 and elapsed > 0:
            fps_value = frame_count / elapsed
            tqdm.write(f"[{vid}] processed {frame_count} frames @ {fps_value:.2f} FPS")
        else:
            tqdm.write(f"[{vid}] no frames processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frame-level features.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = utils.load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
