"""Differentiable rasterisation of a closed contour, for a region-aware loss.

The point-to-point loss cannot see area. A point sliding along the true boundary
is punished even though it changes nothing about the region. The score lives in
the overlap of two areas.

This closes that gap. Occupancy is built the way SoftRas does it: the distance
from a pixel to the polygon is differentiable in the contour points and carries
the gradient, while the inside/outside sign is decided separately and detached -
it is piecewise constant, so it has no useful derivative anyway. A sigmoid over
the signed distance turns the hard indicator into something with a gradient
everywhere, and its temperature sets how far from the boundary a pixel can still
pull on the points near it.

Rasterising happens inside each instance's own region rather than over the whole
image, because objects are small: a grid fine enough for a 3% object would be
absurd at full frame, and one coarse enough for the frame would leave that
object a dozen pixels.
"""

from __future__ import annotations

import numpy as np
import torch


def expand_boxes(boxes: torch.Tensor, factor: float) -> torch.Tensor:
    """Grow boxes about their centre, clamped to the image: ``[M, 4]``.

    The region has to hold wherever the contour ends up, not just where it
    started, or the loss goes blind exactly where the contour overshot.
    """
    centre = torch.stack(
        [(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], dim=1
    )
    half = torch.stack(
        [(boxes[:, 2] - boxes[:, 0]) * 0.5, (boxes[:, 3] - boxes[:, 1]) * 0.5], dim=1
    ) * factor
    grown = torch.cat([centre - half, centre + half], dim=1)
    return grown.clamp(0.0, 1.0)


def expand_boxes_numpy(boxes: np.ndarray, factor: float) -> np.ndarray:
    """``expand_boxes`` for the dataset, which works in numpy.

    Duplicated deliberately rather than routed through torch: the crop and the
    rasteriser must agree exactly, and the cheapest way to be sure of that is a
    shared test rather than a shared tensor library.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    centre = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5,
                       (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1)
    half = np.stack([(boxes[:, 2] - boxes[:, 0]) * 0.5,
                     (boxes[:, 3] - boxes[:, 1]) * 0.5], axis=1) * factor
    return np.clip(np.concatenate([centre - half, centre + half], axis=1), 0.0, 1.0)


def region_grid(rois: torch.Tensor, resolution: int) -> torch.Tensor:
    """Pixel centres of each instance's region: ``[M, R * R, 2]``."""
    device, dtype = rois.device, rois.dtype
    steps = (torch.arange(resolution, device=device, dtype=dtype) + 0.5) / resolution
    ys, xs = torch.meshgrid(steps, steps, indexing="ij")
    unit = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # [R*R, 2]
    origin = rois[:, None, :2]
    size = (rois[:, 2:] - rois[:, :2]).clamp(min=1e-6)[:, None, :]
    return origin + unit[None] * size


def _segment_distance(
    points: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    """Distance from every pixel to every edge: ``[M, G, P]``."""
    edge = ends - starts  # [M, P, 2]
    length = (edge * edge).sum(-1).clamp(min=1e-12)  # [M, P]
    offset = points[:, :, None, :] - starts[:, None, :, :]  # [M, G, P, 2]
    t = ((offset * edge[:, None]).sum(-1) / length[:, None]).clamp(0.0, 1.0)
    nearest = starts[:, None] + t[..., None] * edge[:, None]
    return (points[:, :, None, :] - nearest).norm(dim=-1)


def _inside(
    points: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, edge_mask: torch.Tensor
) -> torch.Tensor:
    """Even-odd ray test, ``[M, G]`` booleans. Deliberately not differentiable.

    A crossing count is an integer; its derivative is zero almost everywhere and
    undefined on the boundary. The gradient the loss needs comes from how far a
    pixel is from the contour, not from which side it is on.
    """
    px = points[..., 0:1]  # [M, G, 1]
    py = points[..., 1:2]
    ax, ay = starts[:, None, :, 0], starts[:, None, :, 1]
    bx, by = ends[:, None, :, 0], ends[:, None, :, 1]

    straddles = (ay > py) != (by > py)
    # x of the edge where it crosses this pixel's row.
    crossing = ax + (py - ay) * (bx - ax) / torch.where(
        (by - ay).abs() < 1e-12, torch.full_like(by, 1e-12), by - ay
    )
    hits = straddles & (px < crossing) & edge_mask[:, None, :]
    return hits.sum(-1) % 2 == 1


def soft_occupancy(
    contours: torch.Tensor,
    counts: torch.Tensor,
    rois: torch.Tensor,
    resolution: int = 64,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Soft inside-ness of every pixel of every region: ``[M, R, R]`` in (0, 1).

    ``temperature`` is in units of the region's own width, so it means the same
    thing for a large object and a small one. Too small and only pixels touching
    the boundary carry gradient; too large and the mask blurs past the shape it
    is meant to describe.
    """
    num_instances, capacity = contours.shape[0], contours.shape[1]
    if num_instances == 0:
        return contours.new_zeros((0, resolution, resolution))

    index = torch.arange(capacity, device=contours.device)[None, :]
    length = counts[:, None].clamp(min=3)
    edge_mask = index < length
    # Each real point joins the next, and the last joins the first: the wrap has
    # to happen at the instance's own count, not at the padded width.
    following = torch.where(index < length - 1, index + 1, torch.zeros_like(index))
    ends = contours.gather(1, following[..., None].expand(-1, -1, 2))

    points = region_grid(rois, resolution)
    distance = _segment_distance(points, contours, ends)
    distance = distance.masked_fill(~edge_mask[:, None, :], float("inf"))
    nearest = distance.min(dim=-1).values  # [M, G]

    with torch.no_grad():
        sign = torch.where(
            _inside(points, contours, ends, edge_mask), 1.0, -1.0
        ).to(nearest.dtype)

    scale = (rois[:, 2] - rois[:, 0]).clamp(min=1e-6)[:, None]
    signed = sign * nearest / (temperature * scale)
    return torch.sigmoid(signed).view(num_instances, resolution, resolution)


def edge_ends(contours: torch.Tensor, counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The far end of each edge and a mask of the real ones.

    Each real point joins the next and the last joins the first, wrapping at the
    instance's own count rather than at the padded width, or an object given 64
    of its 128 slots closes its polygon through the origin.
    """
    index = torch.arange(contours.shape[1], device=contours.device)[None, :]
    length = counts[:, None].clamp(min=3)
    following = torch.where(index < length - 1, index + 1, torch.zeros_like(index))
    ends = contours.gather(1, following[..., None].expand(-1, -1, 2))
    return ends, index < length


def signed_distance(
    points: torch.Tensor, contours: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """Distance from each point to the contour, positive inside: ``[M, G]``.

    The target for the auxiliary head, built here rather than in the dataset for
    two reasons: the contour it is measured against is already on the device, and
    a distance transform per instance per batch would be the most expensive thing
    in the loader. This costs one ``[M, G, P]`` tensor, and G is a coarse probe
    grid rather than the region loss's full raster.
    """
    ends, edge_mask = edge_ends(contours, counts)
    distance = _segment_distance(points, contours, ends)
    distance = distance.masked_fill(~edge_mask[:, None, :], float("inf"))
    nearest = distance.min(dim=-1).values
    sign = torch.where(_inside(points, contours, ends, edge_mask), 1.0, -1.0)
    return sign.to(nearest.dtype) * nearest


def soft_iou(
    occupancy: torch.Tensor, target: torch.Tensor, epsilon: float = 1.0
) -> torch.Tensor:
    """Per-instance soft IoU: ``[M]`` in (0, 1).

    ``epsilon`` is in pixels rather than a token 1e-6, so an instance whose
    region happens to be nearly empty cannot produce an enormous gradient from a
    handful of pixels.
    """
    flat_prediction = occupancy.flatten(1)
    flat_target = target.flatten(1).to(flat_prediction.dtype)
    intersection = (flat_prediction * flat_target).sum(-1)
    union = flat_prediction.sum(-1) + flat_target.sum(-1) - intersection
    return (intersection + epsilon) / (union + epsilon)


def rasterise_into_roi(
    rings, roi, resolution: int, width: float, height: float
) -> np.ndarray:
    """Ground truth for one instance inside its region: ``[R, R]`` uint8.

    Every ring is filled, not just the largest, so the target the region loss
    sees keeps the parts of an occluded object that the single-contour
    representation has to give up. The contour cannot reach them, but it should
    at least not be rewarded for covering the gap between them.
    """
    import cv2

    canvas = np.zeros((resolution, resolution), dtype=np.uint8)
    x0, y0, x1, y1 = [float(v) for v in roi]
    span_x, span_y = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    for ring in rings:
        array = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
        if array.shape[0] < 3:
            continue
        local = np.empty_like(array)
        local[:, 0] = (array[:, 0] / width - x0) / span_x * resolution
        local[:, 1] = (array[:, 1] / height - y0) / span_y * resolution
        cv2.fillPoly(canvas, [np.round(local).astype(np.int32)], 1)
    return canvas
