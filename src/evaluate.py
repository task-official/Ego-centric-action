"""
Evaluation script computing F1@small for detected boundaries.
Writes per-video metrics and an overall average to metrics.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import utils


def match_boundaries(preds: List[int], gts: List[int], tolerance: int) -> Dict[str, int]:
    matched_gt = set()
    tp = 0
    for pred in preds:
        best_idx = None
        best_diff = tolerance + 1
        for idx, gt in enumerate(gts):
            if idx in matched_gt:
                continue
            diff = abs(pred - gt)
            if diff <= tolerance and diff < best_diff:
                best_diff = diff
                best_idx = idx
        if best_idx is not None:
            matched_gt.add(best_idx)
            tp += 1

    fp = len(preds) - tp
    fn = len(gts) - tp
    return {"tp": tp, "fp": fp, "fn": fn}


def safe_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1_small = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "F1_small": f1_small}


def load_predictions(path: Path) -> List[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    tokens = text.split()
    values = []
    for token in tokens:
        try:
            values.append(int(token))
        except ValueError:
            continue
    return sorted(set(values))


def run(cfg: utils.Config) -> None:
    gt_dict = utils.load_ground_truth(cfg.gt_path)
    video_ids = utils.list_video_ids(cfg.data_dir)
    tolerance_frames = max(0, utils.seconds_to_frame(cfg.tolerance_sec, cfg.fps))

    metrics_rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_video_f1: List[float] = []

    for vid in video_ids:
        gt = gt_dict.get(vid, [])
        pred_path = cfg.results_dir / f"{vid}_boundaries.txt"
        preds = load_predictions(pred_path) if pred_path.exists() else []

        counts = match_boundaries(preds, gt, tolerance_frames)
        totals = {k: totals[k] + counts[k] for k in totals}
        scores = safe_metrics(**counts)
        per_video_f1.append(scores["F1_small"])

        metrics_rows.append(
            {
                "video": vid,
                "F1_small": scores["F1_small"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
            }
        )

    overall_scores = safe_metrics(**totals)
    per_video_count = len(per_video_f1) if per_video_f1 else 0
    macro_f1 = sum(per_video_f1) / per_video_count if per_video_count else 0.0
    print(f"Macro F1@small: {macro_f1:.4f}")

    metrics_rows.append(
        {
            "video": "overall",
            "F1_small": overall_scores["F1_small"],
            "precision": overall_scores["precision"],
            "recall": overall_scores["recall"],
            "tp": totals["tp"],
            "fp": totals["fp"],
            "fn": totals["fn"],
        }
    )

    output_path = cfg.results_dir / "metrics.csv"
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["video", "F1_small", "precision", "recall", "tp", "fp", "fn"]
        )
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate boundary detection results.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = utils.load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
