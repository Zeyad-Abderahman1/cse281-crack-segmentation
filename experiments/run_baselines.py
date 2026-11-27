"""Evaluate both baselines on the DEV set and log runs to experiments/log.csv.

Usage:
    python -m experiments.run_baselines --config configs/baseline.json
The test set is deliberately not touched here.
"""
import argparse
import csv
import datetime
import json
import os
import time

import numpy as np

from src.baselines import baseline_simple, baseline_trivial
from src.data import list_pairs, load_image, load_mask, split_dev_test
from src.metrics import iou

LOG_FIELDS = ["run_id", "date", "author", "pipeline", "stage", "method",
              "parameters", "seed", "time_s", "dev_mean_iou"]


def evaluate(fn, pairs, size):
    """Mean IoU of ``fn`` over (image, mask) pairs, plus total seconds."""
    scores, start = [], time.perf_counter()
    for img_path, mask_path in pairs:
        gray = load_image(img_path, gray=True, size=size)
        truth = load_mask(mask_path, size=size)
        scores.append(iou(fn(gray), truth))
    return float(np.mean(scores)), time.perf_counter() - start


def append_log(path, row):
    """Append one run to the experiment CSV, writing a header if new."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.json")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    size = tuple(cfg["size"])
    pairs = list_pairs(cfg["image_dir"], cfg["mask_dir"])
    dev, _test = split_dev_test(pairs, cfg["test_frac"], cfg["seed"])
    print(f"{len(pairs)} pairs -> dev={len(dev)}  (test set left untouched)")

    runs = [("trivial", "global_127", {"thresh": 127}, baseline_trivial),
            ("simple", "otsu+open3x3", {"kernel": 3}, baseline_simple)]
    log_path = cfg["log_path"]
    for i, (name, method, params, fn) in enumerate(runs, start=1):
        score, secs = evaluate(fn, dev, size)
        print(f"{name:8s} {method:14s} dev mean IoU = {score:.4f} ({secs:.1f}s)")
        append_log(log_path, {
            "run_id": f"base-{datetime.date.today():%Y%m%d}-{i}",
            "date": datetime.date.today().isoformat(),
            "author": cfg["author"], "pipeline": f"baseline_{name}",
            "stage": "segmentation", "method": method,
            "parameters": json.dumps(params), "seed": cfg["seed"],
            "time_s": round(secs, 3), "dev_mean_iou": round(score, 4)})


if __name__ == "__main__":
    main()
