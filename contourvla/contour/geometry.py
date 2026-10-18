"""Contour state initialisation and ground-truth correspondence.

Every detected box is turned into an ordered, closed contour of ``P`` points.
That contour is the "robot state" of the policy, and the supervision is the
matching ordered contour on the ground-truth boundary. Because the decoder
predicts one displacement per point, the two contours must be in *one-to-one*
correspondence, and the pairs must be as close as possible so the required
displacements stay inside the range the decoder can express.

All coordinates in this module are normalised to ``[0, 1]`` with respect to the
image, in ``(x, y)`` order.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Initial contour (the detection box -> contour state)
# --------------------------------------------------------------------------- #


def unit_contour(num_points: int, mode: str = "ellipse") -> np.ndarray:
    """Template contour on ``[-1, 1]^2``, ordered by angle and closed.

    ``ellipse`` is the default because a circle is equidistant from the box in
    every direction, so no point starts with a large built-in error, and its
    angular ordering matches the ordering used to align the ground truth.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False, dtype=np.float64)
    if mode == "ellipse":
        return np.stack([np.cos(theta), np.sin(theta)], axis=1)
    if mode in ("rectangle", "octagon"):
        # Project the circle onto the box boundary (rectangle) or onto the
        # octagon inscribed between the box and the diamond.
        cos, sin = np.cos(theta), np.sin(theta)
        scale = np.maximum(np.abs(cos), np.abs(sin))
        rect = np.stack([cos / scale, sin / scale], axis=1)
        if mode == "rectangle":
            return rect
        diamond = np.stack([cos, sin], axis=1) / (np.abs(cos) + np.abs(sin))[:, None]
        return 0.5 * (rect + diamond)
    raise ValueError(f"unknown contour init mode {mode!r}")


def ellipse_perimeter(boxes: np.ndarray) -> np.ndarray:
    """Arc length of the inscribed ellipse, in normalised units: ``[N]``.

    Ramanujan's approximation. It is accurate across the aspect ratios in these datasets.
    This is the quantity the point budget divides, because what makes a contour
    look like a polygon is the distance between consecutive points along the
    boundary - not the object's area, and not its diagonal.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    a = np.abs(boxes[:, 2] - boxes[:, 0]) * 0.5
    b = np.abs(boxes[:, 3] - boxes[:, 1]) * 0.5
    return np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b)))


def point_budget(
    boxes: np.ndarray,
    max_points: int,
    min_points: int,
    spacing_px: float,
    reference_side: int,
    quantum: int = 1,
) -> np.ndarray:
    """How many contour points each box gets: ``[N]`` ints in ``[min, max]``.

    Constant arc spacing, clamped at both ends and rounded to a multiple of
    ``quantum``. Counts are clamped so every instance fits one rectangular
    tensor, while large objects still get more points than small ones.
    """
    perimeter_px = ellipse_perimeter(boxes) * float(reference_side)
    count = perimeter_px / max(spacing_px, 1e-6)
    if quantum > 1:
        count = np.round(count / quantum) * quantum
    return np.clip(np.round(count), min_points, max_points).astype(np.int64)


def boxes_to_contours(boxes: np.ndarray, num_points: int, mode: str = "ellipse") -> np.ndarray:
    """``[N, 4]`` boxes -> ``[N, P, 2]`` initial contours."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    template = unit_contour(num_points, mode)[None]  # [1, P, 2]
    center = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1
    )[:, None, :]
    half = np.stack(
        [(boxes[:, 2] - boxes[:, 0]) * 0.5, (boxes[:, 3] - boxes[:, 1]) * 0.5], axis=1
    )[:, None, :]
    return (center + template * half).astype(np.float32)


def boxes_to_padded_contours(
    boxes: np.ndarray, counts: np.ndarray, capacity: int, mode: str = "ellipse"
) -> tuple[np.ndarray, np.ndarray]:
    """Per-box initial contours at their own point count: ``[N, capacity, 2]``.

    Each row holds ``counts[i]`` points followed by zeros, plus a mask marking
    which are real. The points are spread over the full turn regardless of
    count, so row i is a coarser or finer sampling of the same curve rather than
    a partial one - which is what lets one set of weights serve every budget.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    contours = np.zeros((boxes.shape[0], capacity, 2), dtype=np.float32)
    mask = np.zeros((boxes.shape[0], capacity), dtype=bool)
    for index, count in enumerate(counts):
        count = int(min(max(count, 1), capacity))
        contours[index, :count] = boxes_to_contours(boxes[index : index + 1], count, mode)[0]
        mask[index, :count] = True
    return contours, mask


def smooth_contour(
    points: np.ndarray, harmonics: int = 0, output_points: int = 0
) -> np.ndarray:
    """Band-limit a closed contour and resample it, in one transform.

    A contour is a periodic signal, so treating the outline as ``z = x + iy`` and
    truncating its spectrum is the natural way to smooth it: the result is
    closed and seamless by construction, which a spline fitted through the points
    only is if its boundary conditions are set up for it. Dropping the high
    harmonics removes the point-to-point zigzag that both makes the boundary look
    polygonal and produces most small self-crossings, and evaluating the
    truncated series at more samples than it came with turns the polygon into a
    curve without inventing detail the model did not predict.

    ``harmonics`` of 0 leaves the spectrum alone and only resamples;
    ``output_points`` of 0 keeps the original count.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = points.shape[0]
    if count < 4:
        return points.astype(np.float32)

    length = int(output_points) if output_points else count
    spectrum = np.fft.fft(points[:, 0] + 1j * points[:, 1])
    keep = count // 2 if harmonics <= 0 else min(int(harmonics), count // 2)

    resampled = np.zeros(length, dtype=complex)
    band = min(keep, length // 2)
    resampled[: band + 1] = spectrum[: band + 1]
    if band:
        resampled[-band:] = spectrum[-band:]
    dense = np.fft.ifft(resampled) * (length / count)
    return np.stack([dense.real, dense.imag], axis=-1).astype(np.float32)


def count_self_intersections(points: np.ndarray) -> int:
    """How many times a closed contour crosses itself.

    Not differentiable and not used in training - this is for measuring whether
    the regularisers and the smoothing are working, since "the contour is knotted"
    is otherwise only visible by eye.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = points.shape[0]
    if count < 4:
        return 0
    start = points
    end = np.roll(points, -1, axis=0)

    def side(a, b, c):
        return np.sign(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    crossings = 0
    for i in range(count):
        # Adjacent segments always touch at their shared endpoint, and the first
        # and last are adjacent too, so both are skipped.
        for j in range(i + 2, count - (1 if i == 0 else 0)):
            a, b, c, d = start[i], end[i], start[j], end[j]
            if (
                side(a, b, c) * side(a, b, d) < 0
                and side(c, d, a) * side(c, d, b) < 0
            ):
                crossings += 1
    return crossings


def jitter_boxes(boxes: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Random scale/shift on boxes, in units of the box size.

    Training initialises contours from ground-truth boxes. Jitter keeps the
    decoder from depending on perfectly centred boxes it will not see once the
    detector drives it at test time.
    """
    if amount <= 0.0 or boxes.shape[0] == 0:
        return boxes
    boxes = np.asarray(boxes, dtype=np.float64)
    wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
    offset = rng.uniform(-amount, amount, size=(boxes.shape[0], 4)) * np.concatenate(
        [wh, wh], axis=1
    )
    jittered = boxes + offset
    jittered[:, [0, 2]] = np.sort(jittered[:, [0, 2]], axis=1)
    jittered[:, [1, 3]] = np.sort(jittered[:, [1, 3]], axis=1)
    return np.clip(jittered, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Ground-truth contour extraction
# --------------------------------------------------------------------------- #


def signed_area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def orient_counter_clockwise(polygon: np.ndarray) -> np.ndarray:
    """Fix the traversal direction so cyclic alignment only has to try shifts."""
    return polygon if signed_area(polygon) >= 0 else polygon[::-1].copy()


def resample_closed_polygon(polygon: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a closed polyline to ``num_points`` equally spaced by arc length."""
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if polygon.shape[0] < 2:
        return np.repeat(polygon[:1], num_points, axis=0).astype(np.float32)

    closed = np.concatenate([polygon, polygon[:1]], axis=0)
    segment = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(segment.sum())
    if perimeter < EPS:
        return np.repeat(polygon[:1], num_points, axis=0).astype(np.float32)

    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    targets = np.linspace(0.0, perimeter, num_points, endpoint=False)
    x = np.interp(targets, cumulative, closed[:, 0])
    y = np.interp(targets, cumulative, closed[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def polygons_to_contour(
    polygons: Sequence[Sequence[float]],
    num_points: int,
    width: float,
    height: float,
    mode: str = "polygon",
    raster_side: int = 512,
) -> np.ndarray | None:
    """COCO polygon annotation -> a single ordered outer contour.

    ``polygon`` keeps the largest component, which is what contour-based
    segmenters do for multi-part instances. ``mask`` rasterises the union first
    and traces its outer boundary, which merges parts split by occlusion at the
    cost of a rasterisation per instance.
    """
    rings = []
    for poly in polygons:
        array = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if array.shape[0] >= 3:
            rings.append(array)
    if not rings:
        return None

    if mode == "mask":
        traced = _trace_union_contour(rings, width, height, raster_side)
        if traced is not None:
            rings = [traced]

    ring = max(rings, key=lambda r: abs(signed_area(r)))
    ring = ring / np.array([max(width, EPS), max(height, EPS)])
    ring = orient_counter_clockwise(ring)
    contour = resample_closed_polygon(ring, num_points)
    return np.clip(contour, 0.0, 1.0)


def mask_to_contour(
    binary_mask: np.ndarray, num_points: int, width: float, height: float
) -> np.ndarray | None:
    """Binary mask -> a single ordered outer contour, normalised to ``[0, 1]``.

    Used for gRefCOCO's 1706 crowd annotations, which store run-length encoded
    masks instead of polygons. Without this they would be dropped, and a dropped
    instance is worse than an imprecise one: the box is still emitted by the
    detector, so the instance would sit in the batch with no target.
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return None

    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if largest.shape[0] < 3:
        return None

    scale = np.array([max(binary_mask.shape[1], 1), max(binary_mask.shape[0], 1)], dtype=np.float64)
    ring = orient_counter_clockwise(largest / scale)
    return np.clip(resample_closed_polygon(ring, num_points), 0.0, 1.0)


def rle_to_contour(
    rle: dict, num_points: int, width: float, height: float
) -> np.ndarray | None:
    """Decode a COCO run-length mask and trace its outer boundary."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError:  # pragma: no cover - optional dependency
        return None

    encoded = dict(rle)
    if isinstance(encoded.get("counts"), list):
        encoded = mask_utils.frPyObjects(encoded, int(height), int(width))
    decoded = mask_utils.decode(encoded)
    if decoded.ndim == 3:
        decoded = decoded.max(axis=2)
    if decoded.sum() == 0:
        return None
    return mask_to_contour(decoded, num_points, width, height)


def segmentation_to_contour(
    segmentation,
    num_points: int,
    width: float,
    height: float,
    mode: str = "polygon",
    raster_side: int = 512,
) -> np.ndarray | None:
    """Dispatch on the two COCO segmentation encodings."""
    if segmentation is None:
        return None
    if isinstance(segmentation, dict):
        return rle_to_contour(segmentation, num_points, width, height)
    return polygons_to_contour(segmentation, num_points, width, height, mode, raster_side)


def _trace_union_contour(
    rings: Sequence[np.ndarray], width: float, height: float, raster_side: int
) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dependency
        return None

    scale = raster_side / max(width, height, EPS)
    canvas_w = max(int(round(width * scale)), 2)
    canvas_h = max(int(round(height * scale)), 2)
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(r * scale).astype(np.int32) for r in rings], 1)

    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if largest.shape[0] < 3:
        return None
    return largest / scale


# --------------------------------------------------------------------------- #
# One-to-one correspondence between the initial and ground-truth contours
# --------------------------------------------------------------------------- #


def align_contour(gt: np.ndarray, init: np.ndarray, allow_flip: bool = True) -> np.ndarray:
    """Reorder a closed ground-truth contour to match the initial contour.

    Both contours are cyclic sequences of the same length, so the only freedom
    left after uniform arc-length resampling is the starting index and the
    traversal direction. We pick the combination minimising total squared
    travel, which is the assignment that keeps every point's required
    displacement small.
    """
    gt = np.asarray(gt, dtype=np.float64)
    init = np.asarray(init, dtype=np.float64)
    num_points = init.shape[0]

    candidates = [gt]
    if allow_flip:
        candidates.append(gt[::-1].copy())

    best_cost = np.inf
    best = gt
    index = np.arange(num_points)
    for candidate in candidates:
        # rolled[s, p] = candidate[(p + s) % P]
        rolled = candidate[(index[None, :] + index[:, None]) % num_points]
        cost = ((rolled - init[None]) ** 2).sum(axis=(1, 2))
        shift = int(np.argmin(cost))
        if cost[shift] < best_cost:
            best_cost = float(cost[shift])
            best = rolled[shift]
    return best.astype(np.float32)


def monotone_match_contour(
    gt_dense: np.ndarray,
    init: np.ndarray,
    max_jump: float = 4.0,
    gap_weight: float = 1.0,
    allow_flip: bool = True,
) -> np.ndarray:
    """Assign each initial point a target on the boundary, order preserved.

    Uniform arc-length resampling leaves only a rigid rotation free: point i
    must land on the i-th of P evenly spaced boundary samples. A protrusion
    holding 5% of the perimeter therefore gets 5% of the points however much
    detail it needs, while points on a flat edge beside it barely move. This
    relaxes the rotation into a monotone warp - the targets still go round the
    boundary once, in order, but how much arc each step covers is chosen.

    The objective cannot be travel alone. Points near a protrusion are far from
    its tip, so the cheapest assignment is to crowd them at its base and leap
    over the tip in one step - which is exactly the failure being fixed. A bare
    cap on the step still allows that leap, and it lowers the shape the model
    is trying to reach.

    So skipping boundary is priced rather than merely bounded.
    ``gap_weight`` charges the squared deviation of each step from the uniform
    spacing, in the same arc-length units as the travel it trades against, so 0
    recovers the pure-travel warp and a large value recovers the rigid rotation.
    ``max_jump`` stays as a hard backstop.

    Order preservation matters for its own sake too: a correspondence that
    crosses itself asks two neighbouring points to swap, which is a knot the
    decoder has to tie.
    """
    init = np.asarray(init, dtype=np.float64)
    dense = np.asarray(gt_dense, dtype=np.float64)
    num_points, num_dense = init.shape[0], dense.shape[0]
    if num_points < 3 or num_dense < num_points:
        return align_contour(resample_closed_polygon(dense, num_points), init)

    step = num_dense / num_points
    jump = max(int(round(max_jump * step)), 2)

    best_cost, best = np.inf, None
    for candidate in ([dense, dense[::-1].copy()] if allow_flip else [dense]):
        # Anchor on the best rigid alignment; the warp refines from there rather
        # than searching every start, which would cost a factor of N for a
        # decision the rigid solution already makes well.
        coarse = resample_closed_polygon(candidate, num_points)
        offsets = np.arange(num_points)
        rolled = coarse[(offsets[None, :] + offsets[:, None]) % num_points]
        start = int(np.argmin(((rolled - init[None]) ** 2).sum(axis=(1, 2))))
        anchor = int(round(start * step))

        # Work on a doubled strip so the wrap is a plain forward scan.
        strip = np.concatenate([candidate, candidate], axis=0)[anchor : anchor + num_dense + 1]
        cost = ((strip[None, :, :] - init[:, None, :]) ** 2).sum(-1)  # [P, N+1]

        # Arc length per dense sample, so the gap penalty is in the same units
        # as the squared travel it is traded against.
        perimeter = float(
            np.linalg.norm(candidate - np.roll(candidate, 1, axis=0), axis=1).sum()
        )
        arc = perimeter / max(num_dense, 1)
        uniform = perimeter / num_points
        gaps = np.arange(1, jump + 1)
        gap_cost = gap_weight * (gaps * arc - uniform) ** 2  # [jump]

        table = np.full((num_points, strip.shape[0]), np.inf)
        table[0, 0] = cost[0, 0]
        for i in range(1, num_points):
            # Every reachable predecessor, priced by how far the step skipped.
            previous = table[i - 1]
            options = np.full((jump, previous.shape[0]), np.inf)
            for g in range(1, jump + 1):
                options[g - 1, g:] = previous[:-g] + gap_cost[g - 1]
            table[i] = cost[i] + options.min(axis=0)

        # The last point must leave room to close back onto the anchor.
        tail = table[-1].copy()
        tail[: strip.shape[0] - jump] = np.inf
        end = int(np.argmin(tail))
        if not np.isfinite(tail[end]):
            end = int(np.argmin(table[-1]))
        if tail[end] >= best_cost:
            continue

        path = np.empty(num_points, dtype=np.int64)
        path[-1] = end
        for i in range(num_points - 1, 0, -1):
            low = max(path[i] - jump, 0)
            window = table[i - 1][low : path[i]]
            if not window.size:
                path[i - 1] = 0
                continue
            steps = path[i] - (low + np.arange(window.shape[0]))
            path[i - 1] = low + int(np.argmin(window + gap_cost[steps - 1]))
        best_cost = float(tail[end])
        best = strip[path]

    if best is None:
        return align_contour(resample_closed_polygon(dense, num_points), init)
    return best.astype(np.float32)


def polar_match_contour(gt: np.ndarray, box: np.ndarray, num_points: int) -> np.ndarray:
    """Alternative correspondence: intersect rays from the box centre.

    Guarantees the shortest possible displacement for star-shaped objects but
    degenerates on concave shapes, so it is offered as an option rather than
    the default.
    """
    gt = np.asarray(gt, dtype=np.float64)
    center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
    delta = gt - center
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    order = np.argsort(angles)
    angles, ordered = angles[order], gt[order]

    targets = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    targets = np.where(targets > np.pi, targets - 2.0 * np.pi, targets)

    wrapped_angles = np.concatenate([angles - 2 * np.pi, angles, angles + 2 * np.pi])
    wrapped_points = np.concatenate([ordered, ordered, ordered], axis=0)
    x = np.interp(targets, wrapped_angles, wrapped_points[:, 0])
    y = np.interp(targets, wrapped_angles, wrapped_points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def build_targets(
    gt_contour: np.ndarray,
    init_contour: np.ndarray,
    box: np.ndarray,
    match_mode: str = "arclength_aligned",
) -> np.ndarray:
    if match_mode == "polar":
        return polar_match_contour(gt_contour, box, init_contour.shape[0])
    if match_mode == "arclength_aligned":
        return align_contour(gt_contour, init_contour)
    raise ValueError(f"unknown contour match mode {match_mode!r}")


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #


def contour_to_mask(contour: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterise a normalised contour into a binary mask."""
    import cv2

    points = np.round(contour * np.array([width, height])).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def match_boxes_one_to_one(
    predicted: np.ndarray, truth: np.ndarray, threshold: float = 0.5
) -> int:
    """True positives when each ground truth may be claimed only once.

    Recall alone cannot see over-prediction: let every ground truth pick its
    best match and a model emitting fifty boxes per object scores at least as
    well as one emitting a single box, usually better. Pairing greedily by
    overlap and retiring both sides makes the second box on an object a false
    positive, which is the only way precision means anything here.
    """
    if predicted.shape[0] == 0 or truth.shape[0] == 0:
        return 0
    iou = box_iou(predicted, truth)
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    claimed_predictions: set[int] = set()
    claimed_truths: set[int] = set()
    for prediction, target in order:
        if iou[prediction, target] < threshold:
            break
        if prediction in claimed_predictions or target in claimed_truths:
            continue
        claimed_predictions.add(int(prediction))
        claimed_truths.add(int(target))
    return len(claimed_truths)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``[N, 4]`` and ``[M, 4]`` boxes."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.clip(union, EPS, None)).astype(np.float32)
