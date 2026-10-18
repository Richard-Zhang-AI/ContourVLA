#!/usr/bin/env python3
"""Evaluation under the published GRES protocol, so the numbers are comparable.

`scripts/evaluate.py` reports a `mask_iou` that is useful for tracking training
but is not the benchmark metric:

  1. it skips no-target samples, while gIoU counts them (1 if the prediction is
     empty, 0 otherwise);
  2. it builds the ground truth by tracing one ring per annotation, so
     multi-part instances lose their smaller components;
  3. it rasterises at min(height, 640) rather than the annotation's own
     resolution, and it never computes cIoU.

This script instead follows GRES (arXiv 2306.00968) and InstAlign
(arXiv 2411.15087) exactly:

  gIoU   mean per-sample IoU over *all* samples. For a no-target sample the IoU
         is 1 if the prediction has no foreground pixel, else 0.
  cIoU   total intersection over total union, pooled across the whole split.
  N-acc  of the no-target samples, the share predicted empty.
  T-acc  of the target samples, the share not predicted empty.
  Pr@X   share of samples with IoU > X; a no-target sample counts as a hit iff
         the prediction is empty.

The ground truth is the official annotation decoded with pycocotools at full
image resolution and unioned over every referred instance, which is what the
benchmark means by the referred region. The prediction is the union of the
predicted contours rasterised at the same resolution.

    python scripts/evaluate_gres.py --config configs/rl.yaml \
        --checkpoint outputs/run/final.pt --split testA
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contourvla.config import ContourVLAConfig
from contourvla.contour.geometry import smooth_contour
from contourvla.models.contourvla import ContourVLA
from contourvla.progress import Progress
from contourvla.data.datasets import load_referring_samples


def _decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    """One annotation component, at the image's own resolution."""
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        return mask_utils.decode(rle).astype(bool)
    rles = mask_utils.frPyObjects(segmentation, height, width)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def official_mask(sample) -> np.ndarray:
    """The referred region, decoded from the annotation at full resolution.

    Every component of every referred instance, unioned. This is the target the
    benchmark scores against, and it is deliberately *not* the single-ring
    approximation the training pipeline consumes.
    """
    height, width = int(sample.height), int(sample.width)
    union = np.zeros((height, width), dtype=bool)
    for mask in instance_masks(sample):
        union |= mask
    return union


def instance_masks(sample) -> list[np.ndarray]:
    """One mask per referred instance. Multi-part annotations stay one instance."""
    height, width = int(sample.height), int(sample.width)
    masks = []
    for segmentation in sample.segmentations:
        if segmentation is None:
            continue
        masks.append(_decode_segmentation(segmentation, height, width))
    return masks


def predicted_mask(result, settings, height: int, width: int) -> np.ndarray:
    """Union of the predicted contours, rasterised at the annotation's size."""
    from contourvla.contour.geometry import contour_to_mask

    union = np.zeros((height, width), dtype=bool)
    contours = result.get("contours")
    if contours is None:
        return union
    counts = result.get("point_counts")
    for index, contour in enumerate(contours):
        length = int(counts[index]) if counts is not None else len(contour)
        ring = contour[:length]
        if settings.smooth_harmonics or settings.render_points:
            ring = smooth_contour(ring, settings.smooth_harmonics, settings.render_points)
        union |= contour_to_mask(ring, height, width)
    return union


def predicted_instance_masks(result, settings, height: int, width: int) -> list[np.ndarray]:
    """One mask per predicted contour. Empty contours are not instances."""
    from contourvla.contour.geometry import contour_to_mask

    masks = []
    contours = result.get("contours")
    if contours is None:
        return masks
    counts = result.get("point_counts")
    for index, contour in enumerate(contours):
        length = int(counts[index]) if counts is not None else len(contour)
        ring = contour[:length]
        if settings.smooth_harmonics or settings.render_points:
            ring = smooth_contour(ring, settings.smooth_harmonics, settings.render_points)
        mask = contour_to_mask(ring, height, width)
        if mask.any():
            masks.append(mask)
    return masks


def match_instances(predictions: list[np.ndarray], targets: list[np.ndarray],
                    threshold: float = 0.5) -> int:
    """True positives under greedy one-to-one mask matching at IoU >= threshold."""
    if not predictions or not targets:
        return 0
    iou = np.zeros((len(predictions), len(targets)), dtype=np.float32)
    target_area = [float(mask.sum()) for mask in targets]
    for i, prediction in enumerate(predictions):
        area = float(prediction.sum())
        for j, target in enumerate(targets):
            intersection = float(np.logical_and(prediction, target).sum())
            union = area + target_area[j] - intersection
            if union > 0:
                iou[i, j] = intersection / union
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    claimed_predictions: set[int] = set()
    claimed_targets: set[int] = set()
    for prediction, target in order:
        if iou[prediction, target] < threshold:
            break
        prediction, target = int(prediction), int(target)
        if prediction in claimed_predictions or target in claimed_targets:
            continue
        claimed_predictions.add(prediction)
        claimed_targets.add(target)
    return len(claimed_targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="val", choices=("val", "testA", "testB"))
    # RefCOCO / RefCOCO+ are single-target: every expression refers to exactly
    # one object and there are no no-target samples, so gIoU degenerates to the
    # mIoU those benchmarks report and cIoU to their oIoU - the same arithmetic
    # under different names, which `metric_names` below makes explicit. Nothing
    # else changes; the no-target branches simply never fire.
    parser.add_argument("--dataset", default="",
                        help="override config.data.dataset, e.g. refcoco, refcoco+")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = the whole split")
    # Shuffled before truncating, because the split is ordered by reference and
    # the first samples are wildly unrepresentative: the first 60 of val are 62%
    # no-target against 10.5% for the split.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    config = ContourVLAConfig.load(args.config)
    # GRES counts no-target expressions. A training config that dropped them
    # (some validation loaders do) must not be allowed to drop them here.
    config.data.keep_no_target = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ContourVLA(config).to(device).eval()
    checkpoint_path = os.path.abspath(args.checkpoint) if args.checkpoint else ""
    if checkpoint_path and checkpoint_path != os.path.abspath(config.backbone.checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        print(f"loaded {args.checkpoint} (step {payload.get('step')}), "
              f"{len(missing)} missing / {len(unexpected)} unexpected", flush=True)

    data = copy.deepcopy(config.data)
    data.split = args.split
    if args.dataset:
        data.dataset = args.dataset
    samples = load_referring_samples(data, verbose=True)
    order = np.random.default_rng(args.seed).permutation(len(samples))
    samples = [samples[i] for i in order]
    if args.max_samples:
        samples = samples[: args.max_samples]
    total = len(samples)
    samples = samples[args.shard :: args.num_shards]
    print(f"{args.split}: {len(samples)} of {total} samples "
          f"(shard {args.shard}/{args.num_shards})", flush=True)

    thresholds = (0.5, 0.6, 0.7, 0.8, 0.9)
    ious: list[float] = []
    hits = {t: 0 for t in thresholds}
    total_intersection = total_union = 0.0
    no_target = no_target_correct = 0
    with_target = with_target_correct = 0
    true_positives = false_positives = false_negatives = 0

    # Named per shard, so eight interleaved log files stay attributable.
    bar = Progress(len(samples), prefix=f"[shard {args.shard}] ", interval=200)
    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB")
        with torch.no_grad():
            result = model.predict(image, sample.expression)

        height, width = int(sample.height), int(sample.width)
        prediction = predicted_mask(result, config.contour, height, width)
        empty = not prediction.any()

        if not sample.segmentations:
            # No-target sample: gIoU credits 1 for an empty prediction, 0 else.
            no_target += 1
            no_target_correct += int(empty)
            ious.append(1.0 if empty else 0.0)
            for t in thresholds:
                hits[t] += int(empty)
            # cIoU pools pixels; a no-target sample contributes only whatever
            # foreground was wrongly predicted, which enters the union alone.
            total_union += float(prediction.sum())
            predicted_instances = predicted_instance_masks(
                result, config.contour, height, width
            )
            # No ground-truth instance: every predicted instance is a false positive.
            false_positives += len(predicted_instances)
            bar.update(f"gIoU {np.mean(ious):.4f}")
            continue

        truth = official_mask(sample)
        with_target += 1
        with_target_correct += int(not empty)

        intersection = float(np.logical_and(prediction, truth).sum())
        union = float(np.logical_or(prediction, truth).sum())
        iou = intersection / max(union, 1.0)
        ious.append(iou)
        for t in thresholds:
            hits[t] += int(iou > t)
        total_intersection += intersection
        total_union += union
        targets = instance_masks(sample)
        predicted_instances = predicted_instance_masks(result, config.contour, height, width)
        matched = match_instances(predicted_instances, targets, 0.5)
        true_positives += matched
        false_positives += len(predicted_instances) - matched
        false_negatives += len(targets) - matched

        bar.update(f"gIoU {np.mean(ious):.4f}")

    bar.close(f"gIoU {np.mean(ious):.4f}" if ious else "")

    # Raw accumulators as well as percentages, so shards merge exactly: gIoU is
    # a mean over samples and cIoU is a ratio of pooled pixel counts, neither of
    # which can be recovered from the percentages alone.
    metrics = {
        "dataset": data.dataset,
        "split": args.split,
        # Under a single-target dataset gIoU/cIoU *are* the mIoU/oIoU those
        # benchmarks publish, and N-acc/T-acc are vacuous (no no-target samples,
        # so N-acc has an empty denominator and T-acc counts non-empty
        # predictions). Recorded rather than renamed so the numbers stay
        # traceable to the code that produced them.
        "metric_names": ("gIoU=mIoU, cIoU=oIoU; N-acc/T-acc not applicable"
                         if no_target == 0 else "GRES gIoU / cIoU / N-acc / T-acc"),
        "shard": args.shard,
        "num_shards": args.num_shards,
        "samples": len(ious),
        "gIoU": 100.0 * float(np.mean(ious)) if ious else 0.0,
        "cIoU": 100.0 * total_intersection / max(total_union, 1.0),
        "N-acc": 100.0 * no_target_correct / max(no_target, 1),
        "T-acc": 100.0 * with_target_correct / max(with_target, 1),
        "no_target_samples": no_target,
        "target_samples": with_target,
        "_iou_sum": float(np.sum(ious)),
        "_intersection": total_intersection,
        "_union": total_union,
        "_no_target_correct": no_target_correct,
        "_with_target_correct": with_target_correct,
        "_hits": {str(t): hits[t] for t in thresholds},
        # Instance F1@0.5: greedy one-to-one mask matching, IoU >= 0.5.
        # Micro-averaged over instances. A no-target sample contributes only
        # its false-positive instances.
        "_true_positives": true_positives,
        "_false_positives": false_positives,
        "_false_negatives": false_negatives,
    }
    denom = 2 * true_positives + false_positives + false_negatives
    metrics["F1@0.5"] = 100.0 * (2 * true_positives) / denom if denom else 0.0
    for t in thresholds:
        metrics[f"Pr@{t}"] = 100.0 * hits[t] / max(len(ious), 1)

    print(json.dumps(metrics, indent=2))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
