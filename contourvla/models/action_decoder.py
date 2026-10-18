"""Contour action-chunk decoder.

This is ContourVLA's action expert. It takes the current contour - the analogue
of the robot state - together with the action-ready features produced by the
vision-language interaction module, and emits a displacement for every contour
point. A whole *chunk* of successive displacements is predicted in one
non-autoregressive forward pass; the chunk is then "executed"
(the contour moves), the point features are re-sampled at the new locations, and
the decoder is called again. Detection boxes to object boundary in
``num_calls x chunk_size`` small, bounded steps.

Handling a variable number of instances per image is the reason the decoder is
shaped the way it is. Rather than padding images to a fixed object count, every
box in the batch becomes a row of a flat ``M``-instance tensor carrying its own
memory, so a batch with 1 and 30 objects costs the same per object.

The rows are not quite independent. gRefCOCO expressions can name more than one
object, and neighbouring contours that cannot see each other have no way to
avoid claiming the same pixels. Each memory therefore also holds the instance
tokens of the other objects in the same image, padded to a common width and
masked, tagged with where they sit relative to the object doing the looking.
What crosses over is geometry rather than semantics: a neighbour is told apart
by its position. Only these summary tokens cross between objects; the contour
points never attend outside their own instance, which keeps the cost linear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from ..config import ActionDecoderConfig
from .pixel_decoder import roi_features, sample_points


@dataclass
class ContourPrediction:
    contours: torch.Tensor  # [M, K, T, P, 2] normalised image coords per step
    call_starts: torch.Tensor  # [M, K, P, 2] contour entering each call
    final: torch.Tensor  # [M, P, 2]
    # Router selection stats pooled over every call, for the auxiliary losses
    # and for logging which language-model layer each (decoder layer, call)
    # slot settled on. ``None`` when the router is disabled.
    routing: dict[str, torch.Tensor] | None = None
    action_latents: torch.Tensor | None = None  # [M, K, T, P, 2], detached
    action_log_probs: torch.Tensor | None = None  # [M, K, T, P], joint x/y density


def sample_action_chunk(mean, std, max_step, point_mask, latent=None):
    """Sample/replay a tanh-Gaussian policy in box-relative action units.

    ``mean`` and the saved pre-tanh ``latent`` have shape [M, P, T, 2].
    The sampled value is detached for score-function gradients; replay executes
    this same value, not the updated mean. Padding has neither motion nor density.
    """
    if not math.isfinite(std) or std <= 0 or not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("stochastic actions require finite std > 0 and max_step > 0")
    mean = mean.float()
    mask = point_mask[:, :, None, None]
    mean = torch.where(mask, mean, torch.zeros_like(mean))
    if latent is None:
        latent = mean.detach() + std * torch.randn_like(mean)
    elif latent.shape != mean.shape:
        raise ValueError("saved action shape does not match the decoder chunk")
    latent = torch.where(mask, latent.detach().to(mean), torch.zeros_like(mean))
    normal = torch.distributions.Normal(mean, std)
    log_jacobian = math.log(max_step) + 2 * (
        math.log(2.0) - latent - torch.nn.functional.softplus(-2 * latent)
    )
    log_probs = (normal.log_prob(latent) - log_jacobian).sum(-1)
    log_probs = torch.where(point_mask[:, :, None], log_probs, torch.zeros_like(log_probs))
    return latent, log_probs.permute(0, 2, 1)


def neighbour_index(
    batch_index: torch.Tensor, min_width: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of the other instances sharing each instance's image.

    Returns ``(index, valid)``, both ``[M, W]``. Instances-per-image varies, so
    rows are padded to a common width and ``valid`` marks the real entries.
    ``min_width`` forces a slot to exist even when nothing in the batch has a
    neighbour, which keeps the embedding that marks these tokens attached to
    the graph on every step - DDP runs with ``find_unused_parameters=False``.
    """
    num_instances = batch_index.shape[0]
    same_image = batch_index[:, None] == batch_index[None, :]
    same_image.fill_diagonal_(False)
    counts = same_image.sum(dim=1)
    width = max(int(counts.max()) if num_instances else 0, min_width)
    if width == 0:
        empty = batch_index.new_zeros((num_instances, 0))
        return empty, empty.to(torch.bool)
    # A stable descending sort of the 0/1 mask puts the neighbours first, in
    # ascending row order; the remaining columns are padding.
    order = torch.argsort(same_image.to(torch.uint8), dim=1, descending=True, stable=True)
    index = order[:, :width]
    valid = (
        torch.arange(width, device=batch_index.device)[None, :] < counts[:, None]
    )
    return index, valid


def relative_box_geometry(boxes: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Each neighbour box in the frame of the box looking at it: ``[M, W, 5]``.

    Instance tokens carry *absolute* coordinates - they are pooled over the
    LLM's ``<box><x1><y1><x2><y2></box>`` positions - so on their own they leave
    the decoder to recover "which side, how far, how much bigger" by comparing
    two projected vectors. A neighbour is distinguished by where it is. This
    mirrors the relative coordinates the point queries already get.
    """
    center, size = box_center_size(boxes)
    offset = (center[index] - center[:, None, :]) / size[:, None, :]
    ratio = torch.log(size[index] / size[:, None, :])

    own, other = boxes[:, None, :], boxes[index]
    top_left = torch.maximum(own[..., :2], other[..., :2])
    bottom_right = torch.minimum(own[..., 2:], other[..., 2:])
    overlap = (bottom_right - top_left).clamp(min=0)
    intersection = overlap[..., 0] * overlap[..., 1]
    areas = (own[..., 2] - own[..., 0]) * (own[..., 3] - own[..., 1]) + (
        other[..., 2] - other[..., 0]
    ) * (other[..., 3] - other[..., 1])
    iou = intersection / (areas - intersection).clamp(min=1e-6)
    return torch.cat([offset, ratio, iou.unsqueeze(-1)], dim=-1)


def arc_positions(counts: torch.Tensor, capacity: int) -> torch.Tensor:
    """Where each point sits on its own contour, in turns: ``[M, P]``.

    ``t = i / count`` rather than ``i``, so the same value means the same place
    whether the contour has 64 points or 128. Padded slots run past 1 and wrap,
    which costs nothing because they are masked out of attention and of the loss.
    """
    index = torch.arange(capacity, device=counts.device, dtype=torch.float32)
    return index[None, :] / counts[:, None].clamp(min=1).to(torch.float32)


def fourier_features(values: torch.Tensor, harmonics: int) -> torch.Tensor:
    """Periodic encoding of a position in turns: ``[..., 2 * harmonics]``."""
    orders = torch.arange(1, harmonics + 1, device=values.device, dtype=values.dtype)
    angle = 2.0 * math.pi * values[..., None] * orders
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


def circular_arc_buckets(
    counts: torch.Tensor, capacity: int, num_buckets: int
) -> torch.Tensor:
    """Signed arc distance between every pair of points, bucketed: ``[M, P, P]``.

    Distance is measured the short way round the closed curve and bucketed
    log-spaced per direction, so immediate neighbours are told apart finely and
    the far side of the contour coarsely. Log spacing is what keeps the scale
    meaningful across budgets: neighbours at 1/64 of a turn and neighbours at
    1/128 land in different buckets, so the decoder can see how finely the
    contour it is working on has been sampled.
    """
    positions = arc_positions(counts, capacity)
    delta = positions[:, :, None] - positions[:, None, :]
    delta = delta - torch.round(delta)  # the short way round, in [-0.5, 0.5]

    half = num_buckets // 2
    finest = 1.0 / 1024.0
    magnitude = delta.abs().clamp(min=finest)
    scaled = torch.log(magnitude / finest) / math.log(0.5 / finest)
    step = (scaled.clamp(0.0, 1.0) * (half - 1)).round().long()
    return torch.where(delta >= 0, half + step, half - 1 - step)


def outward_normals(contours: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Unit normal at each point, pointing out of the contour: ``[M, P, 2]``.

    From the chord between the point's two neighbours, which is the local tangent
    of a discretised curve. The neighbours wrap at the instance's own count, not
    at the padded width, or the first and last real point would take their
    tangent from a padding slot.

    The outward sense is fixed from the signed area rather than assumed: the
    initial contours all wind the same way, but nothing stops a contour from
    folding during evolution, and a profile sampled inward on some instances and
    outward on others is a feature the decoder cannot learn a convention for.
    """
    capacity = contours.shape[1]
    index = torch.arange(capacity, device=contours.device)[None, :]
    length = counts[:, None].clamp(min=3)
    # Broadcast [M, 1] against [1, P] to [M, P]; instances have different counts,
    # so the wrap point differs per row.
    ahead = torch.where(index < length - 1, index + 1, torch.zeros_like(index))
    behind = torch.where((index > 0).expand_as(ahead), (index - 1).expand_as(ahead), length - 1)

    def gather(order):
        return contours.gather(1, order[..., None].expand(-1, -1, 2))

    tangent = gather(ahead) - gather(behind)
    normal = torch.stack([tangent[..., 1], -tangent[..., 0]], dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    with torch.no_grad():
        # Shoelace over the real points only; its sign is the winding direction.
        following = gather(ahead)
        area = (
            (contours[..., 0] * following[..., 1] - following[..., 0] * contours[..., 1])
            * (index < length)
        ).sum(-1)
        # For a counter-clockwise ring the shoelace area is positive and
        # (t_y, -t_x) already points out; a clockwise ring needs the flip.
        sense = torch.where(area > 0, 1.0, -1.0).to(normal.dtype)[:, None, None]
    return normal * sense


def evolution_state_features(
    contour: torch.Tensor,
    boxes: torch.Tensor,
    counts: torch.Tensor,
    point_mask: torch.Tensor,
    call: int,
    num_calls: int,
    call_start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-instance contour + evolution-progress state, for the layer router.

    ``LayerRouter.select`` otherwise knows only *which* (decoder layer, call)
    slot is asking - a fixed learned identity - and *which image* is being
    described. It has no way to tell a large, round, nearly-finished contour
    from a small, jagged, freshly-initialised one, or the first call from the
    second. This is what fills that gap: ``[M, router.STATE_DIM]``, box-relative
    throughout so it carries over unchanged across object scale and the
    ``num_points`` allocation.

    Seven numbers, in the order ``router.STATE_DIM`` documents:

    1. log aspect ratio of the box - shape.
    2. log box area - scale.
    3. point budget used / allocated - a cheap complexity proxy (a highly
       faceted or large object is given more of its points to evolve).
    4. contour irregularity - the coefficient of variation of each point's
       radius from the box centre, i.e. how far the current contour already
       is from a perfect ellipse.
    5. call progress, ``call / (num_calls - 1)`` - which stage of the
       evolution this is. Already implicit in the slot index the query
       carries, but explicit here so it can interact multiplicatively with
       the other features rather than only being visible through a frozen
       per-slot vector.
    6 and 7. mean and max normalised displacement the *previous* call actually
       produced (zero on the first call, which has not moved anything yet).
       This is the one genuinely dynamic quantity: it cannot be known ahead of
       running the model, unlike 1-4, and it is what makes routing a function
       of the evolution rather than of the object alone - a call that has
       already travelled most of its budget is a different situation for the
       decoder than one still near its start, whatever the two contours' boxes
       look like.
    """
    center, size = box_center_size(boxes)
    keep = point_mask.to(contour.dtype)
    denom = keep.sum(dim=1, keepdim=True).clamp(min=1.0)

    log_aspect = torch.log(size[:, :1] / size[:, 1:2])
    log_scale = torch.log((size[:, :1] * size[:, 1:2]).clamp(min=1e-6))
    budget = (counts.to(contour.dtype) / contour.shape[1]).unsqueeze(-1)

    relative = (contour - center[:, None, :]) / size[:, None, :]
    radius = relative.norm(dim=-1)
    mean_radius = (radius * keep).sum(dim=1, keepdim=True) / denom
    variance = (((radius - mean_radius) ** 2) * keep).sum(dim=1, keepdim=True) / denom
    irregularity = variance.clamp(min=0).sqrt() / mean_radius.clamp(min=1e-3)

    progress = contour.new_full(
        (contour.shape[0], 1), call / max(num_calls - 1, 1)
    )

    if call_start is not None:
        moved = (contour - call_start) / size[:, None, :]
        moved_norm = moved.norm(dim=-1)
        mean_moved = (moved_norm * keep).sum(dim=1, keepdim=True) / denom
        # Padded slots are masked out of the mean via `keep`, but a raw max
        # would still read them; drive them to -inf first so a fully-padded
        # tail can never win it, then floor the (possibly empty) result at 0.
        max_moved = torch.where(point_mask, moved_norm, moved_norm.new_full((), -1.0))
        max_moved = max_moved.max(dim=1, keepdim=True).values.clamp(min=0.0)
    else:
        mean_moved = contour.new_zeros((contour.shape[0], 1))
        max_moved = contour.new_zeros((contour.shape[0], 1))

    return torch.cat(
        [log_aspect, log_scale, budget, irregularity, progress, mean_moved, max_moved],
        dim=-1,
    )


def sample_normal_profile(
    features: torch.Tensor,
    contours: torch.Tensor,
    counts: torch.Tensor,
    boxes: torch.Tensor,
    batch_index: torch.Tensor,
    feature_sizes: torch.Tensor,
    offsets: torch.Tensor,
    distance_head=None,
    detach_distance: bool = True,
) -> torch.Tensor:
    """Features along each point's normal: ``[M, P, S * C]``.

    A single sample at the point's own location has the receptive field of the
    pixel decoder's convolutions - some tens of pixels - while a point may be
    asked to travel a large fraction of the box. When the boundary lies outside
    that receptive field the point's own feature contains no evidence about
    which way to go, and the only wider view available is the 7x7 region grid in
    the memory, whose cells span tens of pixels themselves. This fills the gap
    between the two, and does it along the one direction the point can move in,
    so choosing a displacement becomes a readout of the profile rather than an
    inference from context.

    Offsets are in units of the box size, so the span means the same thing for a
    large object and a small one.

    With a ``distance_head``, each sample also carries that head's predicted
    distance to the boundary, in units of the box. For a true distance field the
    projection onto the boundary is ``p - sdf(p) * grad sdf(p)``, so a reading
    along the normal turns the displacement from something inferred into
    something the decoder can find at the profile's zero crossing. It costs no
    extra sampling: the head reads the vectors already fetched here.
    """
    normals = outward_normals(contours, counts)
    _, size = box_center_size(boxes)
    step = normals * size[:, None, :]  # [M, P, 2] one unit of offset
    probes = contours[:, :, None, :] + offsets[None, None, :, None] * step[:, :, None, :]

    num_instances, num_points, num_offsets = probes.shape[:3]
    flat = sample_points(
        features,
        probes.reshape(num_instances, num_points * num_offsets, 2).clamp(0.0, 1.0),
        batch_index,
        feature_sizes,
    )
    read = flat.reshape(num_instances, num_points, num_offsets, flat.shape[-1])

    if distance_head is not None:
        predicted = distance_head(read)
        if detach_distance:
            predicted = predicted.detach()
        # Into box units, matching the offsets it sits beside, so the zero
        # crossing and the step the decoder may take are on one scale.
        reach = size.mean(dim=-1).clamp(min=1e-3)[:, None, None]
        read = torch.cat([read, (predicted / reach)[..., None]], dim=-1)

    return read.reshape(num_instances, num_points, -1)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(dim, hidden_dim), nn.GELU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class ChunkDecoderBlock(nn.Module):
    """One invocation of the policy: point queries attend to the object memory."""

    def __init__(self, config: ActionDecoderConfig) -> None:
        super().__init__()
        dim = config.hidden_dim
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=config.nheads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.num_layers)
        # One cross-attention per decoder layer for the routed language context,
        # applied as a residual. The output projection is zeroed so the branch
        # starts as an exact no-op and the model is bit-identical to one built
        # without it; only the projection has gradient at first, which unlocks
        # the rest once it moves. This is the *only* place that gate belongs:
        # ``LayerRouter`` produces the context this consumes, and zeroing both
        # ends of a series pair means neither can ever receive the gradient the
        # other is waiting on - see the note there. Built unconditionally - a few
        # hundred thousand unused parameters cost less than a config-dependent
        # module tree that makes checkpoints incompatible in one direction.
        self.routed_attention = nn.ModuleList(
            nn.MultiheadAttention(dim, config.nheads, dropout=config.dropout,
                                  batch_first=True)
            for _ in range(config.num_layers)
        ) if config.routed_context else None
        for attention in self.routed_attention or []:
            nn.init.zeros_(attention.out_proj.weight)
            nn.init.zeros_(attention.out_proj.bias)
        self.norm = nn.LayerNorm(dim)
        self.head = _mlp(dim, dim, 2 * config.chunk_size, num_layers=3)
        # Start from a near-identity policy: the first updates are small, which
        # keeps the initial contour from being scattered before the features
        # carry any signal.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        memory_padding: torch.Tensor | None = None,
        point_bias: torch.Tensor | None = None,
        routed: list[torch.Tensor] | None = None,
        routed_padding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if routed is None or self.routed_attention is None:
            return self.head(self.norm(self.decoder(
                tgt=queries,
                memory=memory,
                tgt_mask=point_bias,
                memory_key_padding_mask=memory_padding,
            )))

        # Unrolled, because ``nn.TransformerDecoder`` hands the same memory to
        # every layer and the whole point of routing is that each layer gets its
        # own. The loop is what the module does internally; only the extra
        # attention differs per step.
        #
        # A residual, not extra memory tokens. Appending zero-valued tokens is
        # not inert: the attention's key and value projections carry biases, so
        # those slots produce a constant key and value and dilute the softmax
        # over the real memory. With the output projection zero-initialised this
        # branch contributes nothing until it learns to.
        hidden = queries
        for depth, layer in enumerate(self.decoder.layers):
            hidden = layer(
                hidden,
                memory,
                tgt_mask=point_bias,
                memory_key_padding_mask=memory_padding,
            )
            context = routed[depth].to(hidden.dtype)
            attended, _ = self.routed_attention[depth](
                hidden, context, context,
                key_padding_mask=routed_padding,
                need_weights=False,
            )
            hidden = hidden + attended
        if self.decoder.norm is not None:
            hidden = self.decoder.norm(hidden)
        return self.head(self.norm(hidden))


class ContourActionChunkDecoder(nn.Module):
    def __init__(self, config: ActionDecoderConfig, feature_dim: int) -> None:
        super().__init__()
        self.config = config
        dim = config.hidden_dim
        num_points = config.num_points

        self.normal_samples = max(0, config.normal_samples)
        if self.normal_samples > 1:
            span = torch.linspace(-config.normal_span, config.normal_span, self.normal_samples)
        else:
            span = torch.zeros(1)
        self.register_buffer("normal_offsets", span, persistent=False)
        per_sample = feature_dim + (1 if config.sdf_feedback else 0)
        self.point_projection = nn.Sequential(
            nn.Linear(per_sample * span.numel(), dim), nn.LayerNorm(dim)
        )
        self.coord_embedding = _mlp(4, dim, dim)
        # Arc position, not point index: see ``arc_positions`` and the note on
        # ``arc_harmonics``. A table indexed by i would tie the weights to one
        # point budget and would break the contour's periodicity at the seam.
        self.arc_projection = _mlp(2 * config.arc_harmonics, dim, dim)
        self.arc_bias = nn.Parameter(torch.zeros(config.arc_buckets, config.nheads))
        self.call_embedding = nn.Parameter(torch.zeros(config.num_calls, 1, 1, dim))
        self.query_norm = nn.LayerNorm(dim)

        self.instance_projection = nn.Sequential(
            nn.LayerNorm(config.instance_dim), nn.Linear(config.instance_dim, dim)
        )
        self.roi_projection = nn.Sequential(nn.Linear(feature_dim, dim), nn.LayerNorm(dim))
        self.roi_position = nn.Parameter(
            torch.zeros(1, config.roi_output_size**2, dim)
        )
        # Marks a memory token as belonging to a *different* object, so the
        # points can tell "this is me" from "this is the thing next to me",
        # and says where that object sits relative to this one.
        self.other_instance_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.neighbour_geometry = _mlp(5, dim, dim)
        self.memory_norm = nn.LayerNorm(dim)

        num_blocks = 1 if config.share_call_weights else config.num_calls
        self.blocks = nn.ModuleList([ChunkDecoderBlock(config) for _ in range(num_blocks)])

        nn.init.trunc_normal_(self.roi_position, std=0.02)
        nn.init.trunc_normal_(self.call_embedding, std=0.02)
        nn.init.trunc_normal_(self.other_instance_embedding, std=0.02)

    def _block(self, call: int) -> ChunkDecoderBlock:
        return self.blocks[0] if self.config.share_call_weights else self.blocks[call]

    def _build_memory(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        batch_index: torch.Tensor,
        feature_sizes: torch.Tensor,
        instance_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rois = roi_features(
            features,
            boxes,
            batch_index,
            feature_sizes,
            output_size=self.config.roi_output_size,
            expand=self.config.roi_expand,
        )
        rois = rois.flatten(2).transpose(1, 2)  # [M, K*K, C]
        roi_tokens = self.roi_projection(rois) + self.roi_position
        instance = self.instance_projection(instance_features.to(roi_tokens.dtype))

        tokens = [instance.unsqueeze(1)]
        padding = [instance.new_zeros((instance.shape[0], 1), dtype=torch.bool)]
        if self.config.cross_instance:
            index, valid = neighbour_index(batch_index, min_width=1)
            geometry = self.neighbour_geometry(relative_box_geometry(boxes, index))
            tokens.append(instance[index] + self.other_instance_embedding + geometry)
            padding.append(~valid)
        tokens.append(roi_tokens)
        padding.append(roi_tokens.new_zeros(roi_tokens.shape[:2], dtype=torch.bool))

        memory = self.memory_norm(torch.cat(tokens, dim=1))
        mask = torch.cat(padding, dim=1)
        # The own-instance and ROI tokens are never masked, so no row can end
        # up fully padded even when an object has the image to itself.
        return memory, mask if bool(mask.any()) else None

    def _build_queries(
        self,
        features: torch.Tensor,
        contour: torch.Tensor,
        boxes: torch.Tensor,
        batch_index: torch.Tensor,
        feature_sizes: torch.Tensor,
        call: int,
        arc_encoding: torch.Tensor,
        counts: torch.Tensor,
        distance_head=None,
    ) -> torch.Tensor:
        if self.config.sdf_feedback and distance_head is None:
            # Otherwise this surfaces as a linear layer complaining about a
            # width, several frames away from the switch that caused it.
            raise ValueError(
                "sdf_feedback is on but no distance head was passed; the point "
                "projection was sized for the extra channel"
            )
        if self.normal_offsets.numel() > 1:
            point_features = sample_normal_profile(
                features,
                contour,
                counts,
                boxes,
                batch_index,
                feature_sizes,
                self.normal_offsets.to(contour.dtype),
                distance_head=distance_head if self.config.sdf_feedback else None,
                detach_distance=self.config.sdf_detach,
            )
        else:
            point_features = sample_points(features, contour, batch_index, feature_sizes)
        center, size = box_center_size(boxes)
        relative = (contour - center[:, None, :]) / size[:, None, :]
        coords = torch.cat([relative, contour], dim=-1)
        queries = (
            self.point_projection(point_features)
            + self.coord_embedding(coords)
            + arc_encoding
            + self.call_embedding[call]
        )
        return self.query_norm(queries)

    def _point_bias(self, counts: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        """Additive self-attention mask over contour points: ``[M * H, P, P]``.

        Carries two things at once. The learned part is a bias on the arc
        distance between the two points, which is how the decoder is told what
        "adjacent" means at this budget. The infinities hide the padded tail, so
        an object given 64 of its 128 slots attends over 64 points and not over
        64 points plus 64 copies of the top-left corner.
        """
        num_instances, capacity = point_mask.shape
        buckets = circular_arc_buckets(counts, capacity, self.arc_bias.shape[0])
        bias = self.arc_bias[buckets].permute(0, 3, 1, 2)  # [M, H, P, P]

        bias = bias.masked_fill(~point_mask[:, None, None, :], float("-inf"))
        # A query whose every key is masked returns NaN, so each padded point
        # keeps itself. Its output is multiplied out later regardless.
        diagonal = torch.eye(capacity, dtype=torch.bool, device=point_mask.device)
        orphan = diagonal[None, None] & (~point_mask)[:, None, :, None]
        bias = bias.masked_fill(orphan, 0.0)
        return bias.reshape(num_instances * self.config.nheads, capacity, capacity)

    def forward(
        self,
        features: torch.Tensor,
        feature_sizes: torch.Tensor,
        init_contours: torch.Tensor,
        boxes: torch.Tensor,
        batch_index: torch.Tensor,
        instance_features: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        distance_head=None,
        routed=None,
        routed_padding: torch.Tensor | None = None,
        router=None,
        router_context=None,
        router_anneal: float = 1.0,
        action_std: float = 0.0,
        action_latents: torch.Tensor | None = None,
    ) -> ContourPrediction:
        """
        Args:
            features: ``[B, C, H, W]`` pixel-decoder output.
            feature_sizes: ``[B, 2]`` valid feature-map extent per image.
            init_contours: ``[M, P, 2]`` normalised initial contours, padded to
                the allocation; only the first ``point_mask.sum(-1)`` of each are
                real points.
            boxes: ``[M, 4]`` normalised boxes the contours were sampled from.
            batch_index: ``[M]`` image index per instance.
            instance_features: ``[M, D_llm]`` action-ready features.
            point_mask: ``[M, P]`` which allocated points this object evolves.
                All of them when omitted.
            routed: precomputed per-slot memory (one image-level selection,
                shared by both calls). Mutually exclusive with ``router``:
                pass this for a static/legacy routing decision, or ``router``
                for the state-conditioned one described below.
            router: a ``LayerRouter`` (or ``None`` to disable routing). When
                given, routing is decided once *per call* rather than once for
                the whole evolution, each time from ``evolution_state_features``
                of the contour this call is about to act on - so the second
                call's decision can depend on how far the first one actually
                moved things, which no upfront selection could know.
            router_context: ``(context, valid, keys)`` from
                ``router.text_context`` / ``router.keys``, image-level and
                computed once outside this call.
        """
        config = self.config
        num_instances, num_points = init_contours.shape[:2]
        if not math.isfinite(action_std) or action_std < 0:
            raise ValueError("action_std must be finite and nonnegative")
        if action_latents is not None:
            expected = (num_instances, config.num_calls, config.chunk_size, num_points, 2)
            if action_std == 0 or tuple(action_latents.shape) != expected:
                raise ValueError("action replay requires positive std and matching [M,K,T,P,2] latents")
        if num_instances == 0:
            empty = init_contours.new_zeros(
                (0, config.num_calls, config.chunk_size, num_points, 2)
            )
            return ContourPrediction(
                contours=empty,
                call_starts=init_contours.new_zeros((0, config.num_calls, num_points, 2)),
                final=init_contours,
            )

        dtype = features.dtype
        contour = init_contours.to(dtype)
        boxes = boxes.to(dtype)
        _, size = box_center_size(boxes)

        if point_mask is None:
            point_mask = init_contours.new_ones((num_instances, num_points), dtype=torch.bool)
        point_mask = point_mask.to(init_contours.device, torch.bool)
        counts = point_mask.sum(dim=1)

        arc_encoding = self.arc_projection(
            fourier_features(
                arc_positions(counts, num_points), config.arc_harmonics
            ).to(dtype)
        )
        point_bias = self._point_bias(counts, point_mask).to(dtype)
        keep = point_mask[:, :, None].to(dtype)

        memory, memory_padding = self._build_memory(
            features, boxes, batch_index, feature_sizes, instance_features
        )

        per_call: list[torch.Tensor] = []
        starts: list[torch.Tensor] = []
        routing_stats: list[dict[str, torch.Tensor]] = []
        saved_actions, action_log_probs = [], []
        for call in range(config.num_calls):
            starts.append(contour)
            queries = self._build_queries(
                features,
                contour,
                boxes,
                batch_index,
                feature_sizes,
                call,
                arc_encoding,
                counts,
                distance_head,
            )
            # The router gives every (decoder layer, call) pair its own slice
            # of the language model, so the slots are laid out call-major.
            slice_for_call, slice_padding = None, routed_padding
            if router is not None and router_context is not None:
                context, valid, keys = router_context
                # `contour` is what this call is about to act on (`starts[-1]`,
                # just appended); `starts[call - 1]` is where the *previous*
                # call began, so the difference is that call's actual
                # displacement - unknowable before running it, which is why
                # this has to happen per call rather than once up front.
                state = evolution_state_features(
                    contour, boxes, counts, point_mask, call, config.num_calls,
                    call_start=starts[call - 1] if call > 0 else None,
                )
                slice_for_call, slice_padding, stats = router.select(
                    context, valid, keys, batch_index,
                    slot_offset=call * config.num_layers, num_slots=config.num_layers,
                    anneal=router_anneal, state_features=state,
                )
                routing_stats.append(stats)
            elif routed is not None:
                width = len(routed) // config.num_calls
                slice_for_call = routed[call * width : (call + 1) * width]
            chunk = self._block(call)(
                queries, memory, memory_padding, point_bias,
                routed=slice_for_call, routed_padding=slice_padding,
            )
            chunk = chunk.reshape(num_instances, num_points, config.chunk_size, 2)
            if action_std > 0:
                replay = None if action_latents is None else action_latents[:, call].permute(0, 2, 1, 3)
                chunk, log_probs = sample_action_chunk(
                    chunk, action_std, config.max_step, point_mask, replay
                )
                saved_actions.append(chunk.permute(0, 2, 1, 3))
                action_log_probs.append(log_probs)
            # Padded slots stay where they are, so `final` holds the initial
            # contour there instead of an arbitrary drift that a caller which
            # forgot to slice by count would silently draw.
            chunk = torch.tanh(chunk) * config.max_step * keep[..., None]
            # Displacements are in box-relative units; the chunk is executed as
            # a running sum so each step refines the previous one.
            trajectory = chunk.cumsum(dim=2).permute(0, 2, 1, 3)  # [M, T, P, 2]
            contours = contour[:, None] + trajectory * size[:, None, None, :]
            per_call.append(contours)
            contour = contours[:, -1]

        routing = None
        if routing_stats:
            routing = {
                "probabilities": torch.cat([s["probabilities"] for s in routing_stats], dim=1),
                "selection": torch.cat([s["selection"] for s in routing_stats], dim=1),
            }

        return ContourPrediction(
            contours=torch.stack(per_call, dim=1),
            call_starts=torch.stack(starts, dim=1),
            final=contour,
            routing=routing,
            action_latents=torch.stack(saved_actions, dim=1) if saved_actions else None,
            action_log_probs=torch.stack(action_log_probs, dim=1) if action_log_probs else None,
        )


def box_center_size(boxes: torch.Tensor, minimum: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    center = torch.stack(
        [(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], dim=1
    )
    size = torch.stack(
        [boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], dim=1
    ).clamp(min=minimum)
    return center, size
