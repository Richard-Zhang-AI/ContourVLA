"""Segmentation overlays for tracking a run.

A contour-evolution model fails in ways a loss curve hides: contours can
collapse to a point, invert their winding, oscillate between chunk calls, or
track the box instead of the object. All of those are obvious at a glance and
invisible in a scalar, so the trainer renders overlays on a fixed set of samples
at a regular interval.

Two panels are produced per sample:

* **teacher forced** - ground-truth boxes, so the contour branch is shown in
  isolation. Predicted contour against matched ground-truth contour, plus the
  box contour it started from. This is the panel that answers "is the action
  expert learning".
* **generated** - the model's own detections drive the contours, which is the
  end-to-end behaviour. This is the panel that answers "is the whole thing
  working".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from .contour.geometry import smooth_contour

# BGR, chosen to stay legible over natural images.
COLOR_PREDICTION = (64, 255, 64)
COLOR_TARGET = (64, 128, 255)
COLOR_INITIAL = (160, 160, 160)
COLOR_TRAVEL = (200, 120, 255)  # arrows from the initial point to the final one
COLOR_BOX = (0, 200, 255)
COLOR_TEXT = (255, 255, 255)


@dataclass
class Panel:
    image: np.ndarray  # RGB uint8
    caption: str


def _resize(image: np.ndarray, max_side: int) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(
        image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA
    )


def _draw_travel(
    canvas: np.ndarray, start: np.ndarray, end: np.ndarray, stride: int = 4
) -> None:
    """Thin arrows from where each point began to where it ended up.

    The contours alone show the result; these show the policy. A point that
    never left its ellipse, a run of points all piling into the same place, and
    a protrusion the contour failed to reach all look like ordinary boundary
    error in the outline and are obvious here. Every ``stride``-th point is
    drawn, because 128 arrows on one object is a hairball.
    """
    import cv2

    height, width = canvas.shape[:2]
    scale = np.array([width, height], dtype=np.float32)
    for index in range(0, min(len(start), len(end)), stride):
        tail = tuple(np.round(start[index] * scale).astype(int))
        head = tuple(np.round(end[index] * scale).astype(int))
        if abs(head[0] - tail[0]) + abs(head[1] - tail[1]) < 2:
            continue  # a point that barely moved would just be a blob
        cv2.arrowedLine(
            canvas, tail, head, COLOR_TRAVEL, 1, cv2.LINE_AA, tipLength=0.25
        )


def _smoothed(points: np.ndarray, harmonics: int, render: int) -> np.ndarray:
    """The drawn outline, band-limited and densely resampled.

    Drawing the control points directly is what makes a 64-point contour read as
    a polygon; the underlying prediction is unchanged, only how it is shown.
    """
    if harmonics <= 0 and render <= 0:
        return points
    return smooth_contour(points, harmonics, render)


def _draw_contour(canvas: np.ndarray, contour: np.ndarray, colour, thickness: int = 2) -> None:
    import cv2

    height, width = canvas.shape[:2]
    points = np.asarray(contour, dtype=np.float32) * np.array([width, height], dtype=np.float32)
    cv2.polylines(canvas, [points.astype(np.int32)], True, colour, thickness, cv2.LINE_AA)


def _draw_box(canvas: np.ndarray, box: np.ndarray, colour, thickness: int = 2) -> None:
    import cv2

    height, width = canvas.shape[:2]
    corners = (np.asarray(box, dtype=np.float32) * np.array([width, height, width, height])).astype(int)
    cv2.rectangle(canvas, tuple(corners[:2]), tuple(corners[2:]), colour, thickness)


def _draw_caption(canvas: np.ndarray, lines: Sequence[str]) -> None:
    import cv2

    overlay = canvas.copy()
    box_height = 6 + 20 * len(lines)
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas, line[:110], (8, 18 + 20 * index),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA,
        )


def _legend(canvas: np.ndarray, entries: Sequence[tuple[str, tuple]]) -> None:
    import cv2

    y = canvas.shape[0] - 8 - 18 * (len(entries) - 1)
    for label, colour in entries:
        cv2.line(canvas, (10, y - 4), (34, y - 4), colour, 3, cv2.LINE_AA)
        cv2.putText(
            canvas, label, (42, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA
        )
        y += 18


def render_teacher_forced(
    image: Image.Image,
    expression: str,
    predicted: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    valid: np.ndarray,
    max_side: int = 640,
    draw_initial: bool = True,
    point_counts: np.ndarray | None = None,
    smooth_harmonics: int = 0,
    render_points: int = 0,
    draw_travel: bool = True,
    travel_stride: int = 4,
) -> Panel:
    """Overlay for a training sample decoded from ground-truth boxes.

    ``point_counts`` says how many of each row's points are real. Drawing the
    padded tail would run the outline back through the first point, so a large
    object with 128 points and a small one with 64 in the same tensor need
    different slices.
    """
    canvas = _resize(np.array(image.convert("RGB")), max_side).copy()

    def length(index: int, contour: np.ndarray) -> int:
        return int(point_counts[index]) if point_counts is not None else len(contour)

    for index in range(predicted.shape[0]):
        end = length(index, predicted[index])
        if draw_initial:
            _draw_contour(canvas, initial[index][:end], COLOR_INITIAL, 1)
        if draw_travel:
            # Under the contours, so the outlines stay readable on top.
            _draw_travel(canvas, initial[index][:end], predicted[index][:end], travel_stride)
        if index < target.shape[0] and bool(valid[index]):
            _draw_contour(canvas, target[index][:end], COLOR_TARGET, 2)
        _draw_contour(
            canvas,
            _smoothed(predicted[index][:end], smooth_harmonics, render_points),
            COLOR_PREDICTION,
            2,
        )

    error = float("nan")
    if predicted.shape[0] and bool(np.any(valid)):
        selected = np.asarray(valid, dtype=bool)
        error = float(
            np.linalg.norm(predicted[selected] - target[selected], axis=-1).mean()
        )

    _draw_caption(
        canvas,
        [f'"{expression}"', f"teacher forced | {int(predicted.shape[0])} inst | L2 {error:.4f}"],
    )
    legend = [("prediction", COLOR_PREDICTION), ("ground truth", COLOR_TARGET)]
    if draw_travel:
        legend.append((f"travel (every {travel_stride}th)", COLOR_TRAVEL))
    _legend(canvas, legend)
    return Panel(canvas, f"{expression} (teacher forced, L2 {error:.4f})")


def render_generated(
    image: Image.Image,
    expression: str,
    result: dict[str, Any],
    ground_truth: Sequence[np.ndarray] | None = None,
    max_side: int = 640,
    draw_initial: bool = True,
    max_drawn: int = 12,
    smooth_harmonics: int = 0,
    render_points: int = 0,
) -> Panel:
    """Overlay for the model's own detections and contours.

    Early in training the detector emits dozens of spurious boxes, which turns
    the panel into a solid mesh of contours. Only the first ``max_drawn`` are
    drawn; the caption always reports the true count, which is the number that
    matters for spotting the failure.
    """
    canvas = _resize(np.array(image.convert("RGB")), max_side).copy()

    for contour in ground_truth or []:
        _draw_contour(canvas, contour, COLOR_TARGET, 2)

    boxes = np.asarray(result.get("boxes", np.zeros((0, 4))), dtype=np.float32).reshape(-1, 4)
    for box in boxes[:max_drawn]:
        _draw_box(canvas, box, COLOR_BOX, 1)

    contours = result.get("contours")
    initial = result.get("init_contours")
    counts = result.get("point_counts")
    if contours is not None:
        for index, contour in enumerate(np.asarray(contours)[:max_drawn]):
            end = int(counts[index]) if counts is not None else len(contour)
            if initial is not None and index < len(initial):
                if draw_initial:
                    _draw_contour(canvas, initial[index][:end], COLOR_INITIAL, 1)
                _draw_travel(canvas, initial[index][:end], contour[:end])
            _draw_contour(
                canvas,
                _smoothed(contour[:end], smooth_harmonics, render_points),
                COLOR_PREDICTION,
                2,
            )

    detail = f"generated | {boxes.shape[0]} detections"
    if boxes.shape[0] > max_drawn:
        detail += f" (showing {max_drawn})"
    _draw_caption(canvas, [f'"{expression}"', detail])
    _legend(
        canvas,
        [("prediction", COLOR_PREDICTION), ("ground truth", COLOR_TARGET), ("box", COLOR_BOX)],
    )
    return Panel(canvas, f"{expression} (generated, {boxes.shape[0]} det)")


# --------------------------------------------------------------------------- #
# Trainer-facing entry points
# --------------------------------------------------------------------------- #


@torch.no_grad()
def visualize_teacher_forced(
    model,
    batch: dict[str, Any],
    output,
    max_samples: int = 4,
    max_side: int = 640,
    draw_initial: bool = True,
) -> list[Panel]:
    """Render the batch that was just trained on, reusing its forward pass."""
    if output.prediction is None or output.boxes.shape[0] == 0:
        return []

    mask = batch["box_token_mask"].bool()
    counts = mask.sum(dim=1).tolist()
    predicted = output.prediction.final.float().cpu().numpy()
    initial = output.init_contours.float().cpu().numpy()
    targets = batch["target_contours"][mask].float().cpu().numpy()
    valid = batch["contour_valid"][mask].cpu().numpy()
    point_counts = (
        batch["point_counts"][mask].cpu().numpy() if "point_counts" in batch else None
    )

    panels: list[Panel] = []
    offset = 0
    for index, count in enumerate(counts):
        count = int(count)
        if count == 0 or len(panels) >= max_samples:
            offset += count
            continue
        path = batch["image_paths"][index] if "image_paths" in batch else ""
        if not path or not os.path.exists(path):
            offset += count
            continue
        panels.append(
            render_teacher_forced(
                image=Image.open(path),
                expression=batch["expressions"][index],
                predicted=predicted[offset : offset + count],
                target=targets[offset : offset + count],
                initial=initial[offset : offset + count],
                valid=valid[offset : offset + count],
                max_side=max_side,
                draw_initial=draw_initial,
                point_counts=(
                    None if point_counts is None
                    else point_counts[offset : offset + count]
                ),
                smooth_harmonics=model.config.contour.smooth_harmonics,
                render_points=model.config.contour.render_points,
            )
        )
        offset += count
    return panels


@torch.no_grad()
def visualize_generated(
    model,
    samples,
    num_points: int,
    max_samples: int = 4,
    max_side: int = 640,
    draw_initial: bool = True,
) -> list[Panel]:
    """Run full generation on held-out samples and render the result."""
    from .contour.geometry import segmentation_to_contour

    was_training = model.training
    model.eval()
    panels: list[Panel] = []
    for sample in samples[:max_samples]:
        if not os.path.exists(sample.image_path):
            continue
        image = Image.open(sample.image_path).convert("RGB")
        result = model.predict(image, sample.expression)

        truth = []
        for segmentation in sample.segmentations:
            contour = segmentation_to_contour(
                segmentation, num_points, sample.width, sample.height
            )
            if contour is not None:
                truth.append(contour)

        panels.append(
            render_generated(
                image=image,
                expression=sample.expression,
                result=result,
                ground_truth=truth,
                max_side=max_side,
                draw_initial=draw_initial,
                smooth_harmonics=model.config.contour.smooth_harmonics,
                render_points=model.config.contour.render_points,
            )
        )
    if was_training:
        model.train()
    return panels


def pad_to_common_size(panels: Sequence[Panel]) -> list[Panel]:
    """Letterbox a group of panels onto one canvas size.

    COCO images are a mix of portrait and landscape, and wandb refuses to lay
    out a gallery whose members differ in size ("Images sizes do not match"),
    so each group is padded rather than resized - stretching would distort the
    contours the panel exists to show.
    """
    if len(panels) <= 1:
        return list(panels)

    height = max(panel.image.shape[0] for panel in panels)
    width = max(panel.image.shape[1] for panel in panels)
    padded = []
    for panel in panels:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        h, w = panel.image.shape[:2]
        top, left = (height - h) // 2, (width - w) // 2
        canvas[top : top + h, left : left + w] = panel.image
        padded.append(Panel(canvas, panel.caption))
    return padded


def save_panels(panels: Sequence[Panel], directory: str, step: int, prefix: str) -> list[str]:
    import cv2

    if not panels:
        return []
    os.makedirs(directory, exist_ok=True)
    paths = []
    for index, panel in enumerate(panels):
        path = os.path.join(directory, f"step{step:07d}_{prefix}_{index:02d}.jpg")
        cv2.imwrite(path, panel.image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)
    return paths
