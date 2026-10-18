#!/usr/bin/env python3
"""Merge sharded `evaluate_gres.py` outputs into one exact split result.

Percentages cannot be averaged across shards: gIoU is a mean over samples, so it
needs the summed IoU and the sample count, and cIoU is a ratio of pooled pixel
counts. Both are carried in the shard files as raw accumulators for exactly this
reason.

    python scripts/merge_gres_shards.py /tmp/gres_val_rl_*.json
"""

from __future__ import annotations

import json
import sys

THRESHOLDS = ("0.5", "0.6", "0.7", "0.8", "0.9")


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: merge_gres_shards.py <shard json> [...]")

    shards = [json.load(open(p)) for p in paths]
    expected = shards[0]["num_shards"]
    seen = {s["shard"] for s in shards}
    if len(shards) != expected or len(seen) != expected:
        print(f"!! have {sorted(seen)} of {expected} shards; the number below is "
              f"a partial split, not a benchmark result", file=sys.stderr)

    samples = sum(s["samples"] for s in shards)
    iou_sum = sum(s["_iou_sum"] for s in shards)
    intersection = sum(s["_intersection"] for s in shards)
    union = sum(s["_union"] for s in shards)
    no_target = sum(s["no_target_samples"] for s in shards)
    with_target = sum(s["target_samples"] for s in shards)
    no_target_correct = sum(s["_no_target_correct"] for s in shards)
    with_target_correct = sum(s["_with_target_correct"] for s in shards)

    result = {
        "dataset": shards[0].get("dataset", "grefcoco"),
        "split": shards[0]["split"],
        # Single-target splits (RefCOCO / RefCOCO+) carry no no-target samples,
        # so gIoU/cIoU are the mIoU/oIoU those benchmarks publish and N-acc/T-acc
        # do not apply. Repeated here because this is the output that gets quoted.
        "metric_names": shards[0].get("metric_names", ""),
        "shards_merged": len(shards),
        "samples": samples,
        "gIoU": 100.0 * iou_sum / max(samples, 1),
        "cIoU": 100.0 * intersection / max(union, 1.0),
        "N-acc": 100.0 * no_target_correct / max(no_target, 1),
        "T-acc": 100.0 * with_target_correct / max(with_target, 1),
        "no_target_samples": no_target,
        "target_samples": with_target,
    }
    for t in THRESHOLDS:
        hit = sum(s["_hits"][t] for s in shards)
        result[f"Pr@{t}"] = 100.0 * hit / max(samples, 1)

    if "_true_positives" in shards[0]:
        true_positives = sum(s["_true_positives"] for s in shards)
        false_positives = sum(s["_false_positives"] for s in shards)
        false_negatives = sum(s["_false_negatives"] for s in shards)
        denom = 2 * true_positives + false_positives + false_negatives
        result["F1@0.5"] = 100.0 * (2 * true_positives) / denom if denom else 0.0
        result["_true_positives"] = true_positives
        result["_false_positives"] = false_positives
        result["_false_negatives"] = false_negatives

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
