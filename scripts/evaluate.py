#!/usr/bin/env python3
"""Evaluate a ContourVLA checkpoint on a referring-segmentation split."""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contourvla.config import ContourVLAConfig
from contourvla.data.datasets import load_jsonl_samples, load_referring_samples
from contourvla.engine import evaluate
from contourvla.models.contourvla import ContourVLA
from contourvla.visualize import save_panels, visualize_generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--visualize", default="", help="directory for overlay images")
    parser.add_argument("--visualize-count", type=int, default=16)
    args = parser.parse_args()

    config = ContourVLAConfig.load(args.config)
    config.data.split = args.split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContourVLA(config).to(device).eval()

    checkpoint_path = os.path.abspath(args.checkpoint) if args.checkpoint else ""
    if checkpoint_path and checkpoint_path != os.path.abspath(config.backbone.checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        print(f"loaded {args.checkpoint} (step {payload.get('step')})")
        # `strict=False` accepts a checkpoint that omits modules, including one
        # for a different architecture, which leaves those parameters at their
        # initialisation. The frozen backbone is always absent from a training
        # checkpoint, so the warning is restricted to the contour branch.
        trained = [
            name for name in missing
            if name.startswith(("pixel_decoder.", "action_decoder.", "distance_head."))
        ]
        if trained or unexpected:
            print(f"!! {len(trained)} contour-branch parameters were left at their "
                  f"initialisation and {len(unexpected)} in the file went unused.")
            if trained:
                print(f"!!   e.g. {', '.join(sorted(trained)[:3])}")
            print("!!   pass the config the checkpoint was trained with; "
                  "the current one has a different architecture and the number "
                  "below will look plausible anyway.")

    if config.data.dataset == "jsonl":
        samples = load_jsonl_samples(config.data.ref_file, config.data.image_root)
    else:
        samples = load_referring_samples(config.data, verbose=True)
    print(f"{len(samples)} samples in split {args.split!r}")

    metrics = evaluate(model, samples, max_samples=args.max_samples, progress=True)
    print(metrics)

    if args.visualize:
        stride = max(len(samples) // max(args.visualize_count, 1), 1)
        panels = visualize_generated(
            model,
            samples[::stride][: args.visualize_count],
            num_points=config.contour.num_points,
            max_samples=args.visualize_count,
            max_side=config.visualization.max_side,
        )
        paths = save_panels(panels, args.visualize, step=0, prefix="eval")
        print(f"wrote {len(paths)} overlays to {args.visualize}")


if __name__ == "__main__":
    main()
