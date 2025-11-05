"""
Peak detection using min-max normalisation and local maxima selection.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np

import utils


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Scale values to [0, 1] using min-max normalisation."""
    if values.size == 0:
        return values
    v_min = float(values.min())
    v_max = float(values.max())
    if np.isclose(v_max - v_min, 0.0):
        return np.zeros_like(values, dtype=np.float32)
    return (values - v_min) / (v_max - v_min)


def detect_local_maxima(scores: np.ndarray, threshold: Optional[float]) -> List[int]:
    """Select strict local maxima with an optional scalar threshold."""
    peaks: List[int] = []
    if scores.size < 3:
        return peaks
    for idx in range(1, scores.size - 1):
        value = scores[idx]
        if value <= scores[idx - 1] or value <= scores[idx + 1]:
            continue
        if threshold is not None and value <= threshold:
            continue
        peaks.append(idx)
    return peaks


def run(cfg: utils.Config) -> None:
    video_ids = utils.list_video_ids(cfg.data_dir)
    for vid in video_ids:
        dist_path = cfg.results_dir / f"{vid}_distance.npy"
        if not dist_path.exists():
            continue
        raw_scores = np.load(dist_path)
        if raw_scores.size == 0:
            peaks: List[int] = []
        else:
            norm_scores = min_max_normalize(raw_scores.astype(np.float32, copy=False))
            tau = cfg.alpha if cfg.alpha is not None else None
            peaks = detect_local_maxima(norm_scores, tau)

        center_offset = cfg.window // 2
        boundaries = [int(idx * cfg.stride + center_offset) for idx in peaks]
        line = " ".join([vid] + [str(b) for b in boundaries])
        out_path = cfg.results_dir / f"{vid}_boundaries.txt"
        out_path.write_text(line, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Peak detection via local maxima.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = utils.load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
