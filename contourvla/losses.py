"""Training objectives: box-token cross-entropy plus contour regression."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import LossConfig
from .contour.raster import signed_distance, soft_iou, soft_occupancy
from .models.action_decoder import ContourPrediction, box_center_size


def zero_touch(module: nn.Module) -> torch.Tensor:
    """A scalar that is always exactly zero but depends on every parameter.

    When an image yields no usable instance the contour branch is skipped, so
    its parameters receive no gradient and DDP stalls waiting for buckets that
    never arrive. Adding this term keeps every parameter in the graph without
    changing the loss value.
    """
    total = None
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        term = parameter.sum() * 0.0
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), requires_grad=True)
    return total


def circular_neighbour(counts: torch.Tensor, capacity: int, shift: int) -> torch.Tensor:
    """Index of each point's neighbour ``shift`` steps round its own contour.

    ``torch.roll`` cannot be used once budgets vary: it wraps at the allocation,
    so on a 64-point contour padded to 128 the point before index 0 comes out as
    index 127 - a padding slot - instead of index 63. Anything that looks at
    neighbours has to wrap at ``counts`` instead, which is what this gives.
    Padded slots map to themselves and are masked out by the caller.
    """
    index = torch.arange(capacity, device=counts.device)[None, :]
    length = counts[:, None].clamp(min=1)
    return torch.where(index < length, (index + shift) % length, index)


def gather_points(points: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``[M, P, 2]`` reordered by ``[M, P]`` indices."""
    return points.gather(1, index[..., None].expand(-1, -1, points.shape[-1]))


def chunk_targets(
    prediction: ContourPrediction, target_contours: torch.Tensor
) -> torch.Tensor:
    """Per-step supervision for one action chunk.

    The final step of the last call must reach the ground truth. Intermediate
    steps are supervised towards a linear interpolation between the contour that
    entered the call and the ground truth, which spreads the total displacement
    evenly over the chunk instead of asking the first step to do all the work.

    Returns ``[M, K, T, P, 2]``.
    """
    num_instances, num_calls, chunk_size = prediction.contours.shape[:3]
    steps = torch.arange(1, chunk_size + 1, device=target_contours.device)
    fraction = (steps / chunk_size).view(1, 1, chunk_size, 1, 1)
    start = prediction.call_starts.detach().unsqueeze(2)  # [M, K, 1, P, 2]
    goal = target_contours[:, None, None]  # [M, 1, 1, P, 2]
    return start + fraction * (goal - start)


class ContourVLALoss(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def segmentation_loss(
        self,
        prediction: ContourPrediction,
        target_contours: torch.Tensor,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        region_boxes: torch.Tensor | None = None,
        region_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Smooth-L1 between predicted and matched ground-truth contour points.

        The residual is expressed in units of the box size so that a small
        object and a large one contribute equally, and so the numeric range of
        the loss is comparable across images - which is what makes a fixed
        fallback value meaningful when there is nothing to segment.

        Points are averaged over the ones the instance actually evolves. Taking
        the mean over the whole allocation instead would quietly scale every
        small object's loss down by the fraction of slots it left empty, and the
        objects with the fewest points are the ones already hardest to fit.
        """
        config = self.config
        predicted = prediction.contours
        target = chunk_targets(prediction, target_contours)

        if config.normalize_by_box:
            _, size = box_center_size(boxes.to(predicted.dtype))
            scale = size.view(-1, 1, 1, 1, 2)
        else:
            scale = torch.ones_like(predicted[:1, :1, :1, :1])

        residual = (predicted - target.to(predicted.dtype)) / scale
        per_point = F.smooth_l1_loss(
            residual, torch.zeros_like(residual), beta=config.smooth_l1_beta, reduction="none"
        ).sum(dim=-1)  # [M, K, T, P]

        num_calls, chunk_size = per_point.shape[1], per_point.shape[2]
        step = torch.arange(
            1, num_calls * chunk_size + 1, device=per_point.device, dtype=per_point.dtype
        )
        weight = (step / step[-1]) ** config.step_weight_power
        weight = (weight / weight.sum()).view(1, num_calls, chunk_size, 1)

        if point_mask is None:
            point_mask = per_point.new_ones(per_point.shape[0], per_point.shape[-1])
        keep = point_mask.to(per_point.dtype)
        points = keep.sum(dim=1).clamp(min=1)  # [M]

        if config.travel_weight > 0:
            # A protrusion is a few points that have to travel several times
            # further than the rest of the contour, and averaging over points
            # buries them: they are outvoted by the majority that barely has to
            # move, so they are the ones that never arrive. Weighting by the
            # distance each point is being asked to cover puts the gradient back
            # where the work is. The weight comes from the data, not from the
            # current error, so it does not chase the model's own mistakes.
            start = prediction.call_starts[:, 0].detach()
            travel = (
                (target_contours.to(predicted.dtype) - start) / scale[:, 0, 0]
            ).norm(dim=-1)  # [M, P]
            average = (travel * keep).sum(1, keepdim=True) / points[:, None]
            weight_by_travel = 1.0 + config.travel_weight * (
                travel / average.clamp(min=1e-6)
            )
            # Renormalised so this redistributes the loss rather than scaling it,
            # which would silently change the balance against detection.
            weight_by_travel = weight_by_travel / (
                (weight_by_travel * keep).sum(1, keepdim=True) / points[:, None]
            ).clamp(min=1e-6)
            per_point = per_point * weight_by_travel[:, None, None, :]

        per_point = per_point * keep[:, None, None, :]
        instance_loss = (per_point * weight).sum(dim=(1, 2)).sum(dim=-1) / points  # [M]
        instance_loss = instance_loss * valid.to(instance_loss.dtype)
        denominator = valid.sum().clamp(min=1).to(instance_loss.dtype)
        loss = instance_loss.sum() / denominator

        counts = keep.sum(dim=1).long()
        capacity = keep.shape[1]
        if config.smoothness_weight > 0 or config.spacing_weight > 0:
            final = prediction.final / scale.squeeze(1).squeeze(1)
            ahead = gather_points(final, circular_neighbour(counts, capacity, 1))
            behind = gather_points(final, circular_neighbour(counts, capacity, -1))

        if config.smoothness_weight > 0:
            # Distance from the midpoint of the two neighbours: zero on a
            # straight run, large on a spike. Spikes are where a contour crosses
            # itself, so this is the cheapest thing that discourages knots.
            curvature = final - 0.5 * (ahead + behind)
            smooth = (curvature.abs().sum(-1) * keep).sum(-1) / points
            loss = loss + config.smoothness_weight * (
                smooth * valid.to(curvature.dtype)
            ).sum() / denominator

        if config.spacing_weight > 0:
            # Points bunching up leaves the rest of the boundary to be spanned by
            # a few long chords, and the bunched ones are free to swap order.
            # Penalising the spread of chord lengths keeps the parameterisation
            # near uniform without prescribing what the shape should be.
            chord = (ahead - final).norm(dim=-1) * keep
            mean_chord = chord.sum(-1, keepdim=True) / points[:, None]
            spread = (((chord - mean_chord) * keep) ** 2).sum(-1) / points
            loss = loss + config.spacing_weight * (
                spread * valid.to(chord.dtype)
            ).sum() / denominator

        metrics: dict[str, float] = {}
        if config.region_weight > 0 and region_boxes is not None:
            # The term the model is actually scored on. Everything above moves
            # points towards their assigned targets; this one moves the area they
            # enclose towards the area it should cover, which is what mask IoU
            # measures and what a missed protrusion costs.
            # Scored at the end of every call, not only the last. The point term
            # already supervises both calls, towards a target interpolated from
            # where each call started, but the region term saw the final state
            # alone - so the first call was told where its points should go and
            # never what area they should end up enclosing, despite its output
            # being the second call's starting contour. Weighted so the last call
            # still dominates, the same way the point term ramps across steps.
            states = list(prediction.call_starts.unbind(1))[1:] + [prediction.final]
            weights = torch.linspace(
                1.0 / len(states), 1.0, len(states), device=prediction.final.device
            )
            region = prediction.final.new_zeros(())
            for index, (state, weight) in enumerate(zip(states, weights)):
                occupancy = soft_occupancy(
                    state,
                    counts,
                    region_boxes.to(state.dtype),
                    resolution=region_masks.shape[-1],
                    temperature=config.soft_iou_temperature,
                )
                overlap = soft_iou(occupancy, region_masks)
                region = region + weight * (
                    (1.0 - overlap) * valid.to(overlap.dtype)
                ).sum() / denominator
                if index == len(states) - 1:
                    metrics["soft_iou"] = float(
                        (overlap.detach() * valid).sum() / denominator
                    )
            region = region / float(weights.sum())
            loss = loss + config.region_weight * region
            metrics["loss_region"] = float(region.detach())

        with torch.no_grad():
            distance = (
                (prediction.final - target_contours.to(predicted.dtype)) / scale[:, 0, 0]
            ).norm(dim=-1)
            final_error = (distance * keep).sum(dim=-1) / points
            metrics["seg_final_error"] = float((final_error * valid).sum() / denominator)
        return loss, metrics

    def forward(
        self,
        detection_loss: torch.Tensor | None,
        prediction: ContourPrediction | None,
        target_contours: torch.Tensor | None,
        boxes: torch.Tensor | None,
        valid: torch.Tensor | None,
        contour_modules: nn.Module,
        point_mask: torch.Tensor | None = None,
        region_boxes: torch.Tensor | None = None,
        region_masks: torch.Tensor | None = None,
        distance: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        config = self.config
        metrics: dict[str, float] = {}

        device = next(contour_modules.parameters()).device
        total = torch.zeros((), device=device, dtype=torch.float32)

        if detection_loss is not None:
            total = total + config.detection_weight * detection_loss.float()
            metrics["loss_det"] = float(detection_loss.detach())

        has_instances = (
            prediction is not None
            and valid is not None
            and prediction.contours.shape[0] > 0
            and bool(valid.any())
        )
        # Always reported, so the logged average is the fraction of steps that
        # fell back rather than a constant 1.0 over the steps that did.
        metrics["seg_fallback"] = 0.0
        if has_instances:
            seg_loss, seg_metrics = self.segmentation_loss(
                prediction, target_contours, boxes, valid, point_mask,
                region_boxes, region_masks,
            )
            total = total + config.segmentation_weight * seg_loss.float()
            metrics["loss_seg"] = float(seg_loss.detach())
            metrics.update(seg_metrics)
            metrics["num_instances"] = float(valid.sum())
            if point_mask is not None:
                # Logged so the budget rule is visible in training curves rather
                # than inferred from the config.
                counts = point_mask.sum(dim=1).to(torch.float32)
                kept = valid.to(counts.dtype)
                metrics["points_per_instance"] = float(
                    (counts * kept).sum() / kept.sum().clamp(min=1)
                )
        else:
            # Nothing to trace: hold the segmentation term at a constant inside
            # the range a real loss occupies, so the total stays on scale and
            # optimiser statistics are not disturbed by an image that happens to
            # contain no object.
            fallback = torch.as_tensor(
                config.segmentation_fallback, device=device, dtype=torch.float32
            )
            total = total + config.segmentation_weight * (
                fallback + zero_touch(contour_modules).float()
            )
            metrics["loss_seg"] = float(config.segmentation_fallback)
            metrics["seg_fallback"] = 1.0
            metrics["num_instances"] = 0.0

        if distance is not None and has_instances:
            sdf, sdf_metrics = self.distance_loss(
                distance, target_contours, boxes, valid, point_mask
            )
            total = total + config.sdf_weight * sdf.float()
            metrics.update(sdf_metrics)

        metrics["loss"] = float(total.detach())
        return total, metrics

    def distance_loss(
        self,
        distance: tuple[torch.Tensor, torch.Tensor],
        target_contours: torch.Tensor,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        point_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Auxiliary: the features must say how far the boundary is.

        Truncated, because past the profile's reach the distance can be neither
        read nor acted on, and untruncated far values would dominate an L1 while
        carrying nothing. Everything is divided by the instance's box first, so a
        large object and a small one contribute the same way.

        ``sdf_r2`` is reported because it, and not the loss, is the number that
        decides whether the head is good enough to feed back into the decoder;
        see ActionDecoderConfig.sdf_feedback.
        """
        config = self.config
        locations, predicted = distance
        counts = (
            point_mask.sum(dim=1)
            if point_mask is not None
            else torch.full(
                (target_contours.shape[0],),
                target_contours.shape[1],
                device=target_contours.device,
            )
        )
        with torch.no_grad():
            truth = signed_distance(
                locations.to(target_contours.dtype), target_contours, counts
            )

        _, size = box_center_size(boxes.to(predicted.dtype))
        reach = size.mean(dim=-1).clamp(min=1e-3)[:, None]
        limit = config.sdf_truncation
        # Only the target is truncated. Clamping the prediction too is the usual
        # formulation and it is a trap: the derivative of a clamp outside its
        # range is exactly zero, so a head whose outputs happen to initialise
        # beyond the band receives no gradient at all and never recovers.
        scaled = predicted / reach
        target = (truth.to(predicted.dtype) / reach).clamp(-limit, limit)

        keep = valid.to(predicted.dtype)[:, None]
        denominator = keep.sum().clamp(min=1) * predicted.shape[1]
        error = torch.nn.functional.smooth_l1_loss(
            scaled, target, beta=config.smooth_l1_beta, reduction="none"
        )
        loss = (error * keep).sum() / denominator

        with torch.no_grad():
            kept = keep.expand_as(target) > 0
            live = target[kept]
            residual = ((scaled[kept] - live) ** 2).sum()
            spread = ((live - live.mean()) ** 2).sum()
            metrics = {
                "loss_sdf": float(loss.detach()),
                "sdf_r2": float(1.0 - residual / spread.clamp(min=1e-9)),
            }
        return loss, metrics
