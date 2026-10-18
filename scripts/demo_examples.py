#!/usr/bin/env python3
"""Run the released checkpoint on the bundled example images."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contourvla.config import ContourVLAConfig
from contourvla.models.contourvla import ContourVLA
from contourvla.visualize import render_generated, save_panels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rl.yaml")
    parser.add_argument("--checkpoint", default="ckpts/contourvla.pt")
    parser.add_argument("--manifest", default="asset/examples.json")
    parser.add_argument("--output", default="outputs/examples")
    parser.add_argument("--max-side", type=int, default=960)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = (root / args.manifest).resolve()
    output_dir = (root / args.output).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    examples = manifest.get("examples", [])
    if not examples:
        raise ValueError(f"manifest has no examples: {manifest_path}")

    config = ContourVLAConfig.load(str((root / args.config).resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContourVLA(config).to(device).eval()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    checkpoint_path = str(checkpoint.resolve())
    missing, unexpected = [], []
    if checkpoint_path != os.path.abspath(config.backbone.checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    trained_missing = [
        name for name in missing
        if name.startswith(("pixel_decoder.", "action_decoder.", "distance_head.", "router."))
    ]
    if trained_missing or unexpected:
        raise RuntimeError(
            "checkpoint/config mismatch: "
            f"{len(trained_missing)} trained parameters missing, "
            f"{len(unexpected)} unexpected parameters"
        )

    panels = []
    records = []
    for item in examples:
        image_path = (manifest_path.parent / item["image"]).resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        expression = str(item["expression"])
        with torch.inference_mode():
            result = model.predict(image, expression)
        panels.append(
            render_generated(
                image=image,
                expression=expression,
                result=result,
                max_side=args.max_side,
                smooth_harmonics=config.contour.smooth_harmonics,
                render_points=config.contour.render_points,
            )
        )
        records.append(
            {
                "id": item.get("id", image_path.stem),
                "image": item["image"],
                "expression": expression,
                "detections": int(len(result.get("boxes", []))),
                "answer": result.get("answer", ""),
            }
        )

    paths = save_panels(panels, str(output_dir), step=0, prefix="example")
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"checkpoint": str(Path(args.checkpoint)), "examples": records}, handle, indent=2)
    print(f"wrote {len(paths)} overlays to {output_dir}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
