"""
Compute subspace-based change magnitudes for consecutive time windows.
For each sliding window we form a PCA subspace, compare adjacent subspaces via
the Grassmannian difference magnitude, and emit a boundary score sequence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

import utils


def compute_subspace_signal(
    features: np.ndarray,
    window: int,
    stride: int,
    rank: int,
    device: torch.device,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute a 1D signal of subspace distances across the feature sequence.
    Returns an array with length equal to number of frames; entries are zero
    except at stride positions where the distance is recorded.
    """
    num_frames = features.shape[0]
    if num_frames < window or window <= 0 or stride <= 0:
        return np.zeros(0, dtype=np.float32)

    starts = list(range(0, num_frames - window + 1, stride))
    if len(starts) < 2:
        return np.zeros(0, dtype=np.float32)

    projectors: List[Optional[np.ndarray]] = []
    for start in starts:
        window_feat = features[start : start + window].astype(np.float32, copy=False)
        centered = window_feat - window_feat.mean(axis=0, keepdims=True)
        if normalize:
            std = centered.std(axis=0, keepdims=True)
            std[std == 0] = 1.0
            centered = centered / std
        cov = centered.T @ centered
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            projectors.append(None)
            continue
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        sub_rank = min(rank, eigvecs.shape[1])
        if sub_rank == 0:
            projectors.append(None)
            continue
        basis = eigvecs[:, :sub_rank]
        projectors.append(basis @ basis.T)

    # 距離列は隣接窓の差のみ: 例 T=90, W=20, S=5 → len(distance)=14
    feat_dim = features.shape[1]
    eye = np.eye(feat_dim, dtype=np.float32)
    distances = np.zeros(len(projectors) - 1, dtype=np.float32)
    for idx in range(len(projectors) - 1):
        P = projectors[idx]
        P_next = projectors[idx + 1]
        if P is None or P_next is None:
            continue
        G = P + P_next
        try:
            eigvals = np.linalg.eigh(G)[0]
        except np.linalg.LinAlgError:
            continue
        mask = (eigvals > 1e-6) & (eigvals < 1.0 - 1e-6)
        distances[idx] = float(np.count_nonzero(mask))

    return distances


def run(cfg: utils.Config) -> None:
    video_ids = utils.list_video_ids(cfg.data_dir)
    for vid in video_ids:
        feat_path = cfg.features_dir / f"{vid}.npy"
        if not feat_path.exists():
            continue
        features = np.load(feat_path)
        start_time = time.perf_counter()
        signal = compute_subspace_signal(
            features, cfg.window, cfg.stride, cfg.pca_rank, cfg.device
        )
        elapsed = time.perf_counter() - start_time
        output_path = cfg.results_dir / f"{vid}_distance.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, signal)
        meta_path = cfg.results_dir / f"{vid}_meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        meta["boundary_inference_time_sec"] = elapsed
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute subspace distance signals.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = utils.load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
