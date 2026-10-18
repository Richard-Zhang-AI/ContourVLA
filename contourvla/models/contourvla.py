"""ContourVLA: open-vocabulary segmentation by contour evolution.

The embedded multimodal module predicts boxes and image features. A
contour action decoder then refines each box into an object mask. The model is
trained with detection and contour regression objectives.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

import math

import numpy as np
import torch
from torch import nn

from ..config import ContourVLAConfig
from ..contour.geometry import boxes_to_padded_contours, point_budget, resample_closed_polygon
from ..contour.raster import expand_boxes, region_grid
from ..tokens import (
    build_prompt,
    decode_box_spans,
    find_box_token_spans,
    parse_boxes,
    token_span_for_chars,
)
from .action_decoder import ContourActionChunkDecoder, ContourPrediction
from .backbone import BackboneOutput, VLInteractionModule, load_checkpoint_payload
from .pixel_decoder import DistanceHead, PixelDecoder, sample_points
from .router import LayerRouter, routing_losses


@dataclass
class ContourVLAOutput:
    backbone: BackboneOutput
    prediction: ContourPrediction | None
    boxes: torch.Tensor  # [M, 4]
    init_contours: torch.Tensor  # [M, P, 2]
    batch_index: torch.Tensor  # [M]
    feature_sizes: torch.Tensor
    point_mask: torch.Tensor | None = None  # [M, P], which points are real
    # ``(locations, predicted)`` from the auxiliary distance head, training only.
    distance: tuple[torch.Tensor, torch.Tensor] | None = None
    # Router selection probabilities, for the auxiliary losses and for logging
    # which language-model layer each decoder layer settled on.
    routing: dict | None = None


class ContourVLA(nn.Module):
    def __init__(self, config: ContourVLAConfig) -> None:
        super().__init__()
        self.config = config
        payload = load_checkpoint_payload(config.backbone.checkpoint_path)
        self.backbone = VLInteractionModule(config.backbone, payload=payload)

        config.action_decoder.instance_dim = self.backbone.hidden_size
        config.action_decoder.routed_context = config.router.enabled
        self.pixel_decoder = PixelDecoder(config.pixel_decoder, self.backbone.hidden_size)
        self.action_decoder = ContourActionChunkDecoder(
            config.action_decoder, feature_dim=config.pixel_decoder.hidden_dim
        )
        self.router = None
        if config.router.enabled:
            depth = self.backbone.num_layers
            candidates = tuple(range(config.router.stride - 1, depth, config.router.stride))
            self.backbone.extra_taps = set(candidates)
            self.router = LayerRouter(
                candidates=candidates,
                llm_dim=self.backbone.hidden_size,
                dim=config.action_decoder.hidden_dim,
                slots=config.action_decoder.num_layers * config.action_decoder.num_calls,
                key_dim=config.router.key_dim,
                max_tokens=config.router.max_tokens,
                prior_scale=config.router.prior_scale,
                state_conditioning=config.router.state_conditioning,
            )

        # Built when either the loss or the feedback path wants it, so switching
        # `sdf_feedback` on without a weight is not silently a no-op head.
        wanted = config.loss.sdf_weight > 0 or config.action_decoder.sdf_feedback
        self.distance_head = (
            DistanceHead(config.pixel_decoder.hidden_dim, config.loss.sdf_hidden)
            if wanted
            else None
        )
        missing, unexpected = self.load_state_dict(payload["model"], strict=False)
        if missing or unexpected:
            raise ValueError(
                "checkpoint/config mismatch: "
                f"{len(missing)} missing parameters, {len(unexpected)} unexpected parameters"
            )
        del payload

    # ------------------------------------------------------------------ #
    # module groups
    # ------------------------------------------------------------------ #

    @property
    def contour_modules(self) -> nn.Module:
        """Everything downstream of the backbone, as one module for the loss.

        Two things read this and both are load-bearing: ``parameter_groups``
        gives whatever is in here the head learning rate, and ``zero_touch``
        walks it to keep every parameter attached to the graph on a step with no
        instance to segment. A module missing from here is therefore silently
        never trained *and* a DDP crash waiting for an image with no target.

        Built fresh per call rather than cached on ``self``: assigning an
        ``nn.ModuleList`` would register these modules a second time, duplicating
        every one of their tensors in ``state_dict``, and doing it lazily would
        grow the module tree on the first backward - after DDP has already taken
        its parameter list from it. The list only holds references, so rebuilding
        it costs nothing.
        """
        parts = [self.pixel_decoder, self.action_decoder]
        if self.router is not None:
            parts.append(self.router)
        if self.distance_head is not None:
            parts.append(self.distance_head)
        return nn.ModuleList(parts)

    def parameter_groups(self, lr: float, backbone_lr: float, weight_decay: float):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.contour_modules.parameters() if p.requires_grad]
        groups = []
        if backbone_params:
            groups.append(
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay}
            )
        if head_params:
            groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})
        return groups

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #

    def forward(self, batch: Mapping[str, Any]) -> ContourVLAOutput:
        device = self.backbone.device
        pixel_values = batch["pixel_values"].to(device)
        image_grid_hws = batch["image_grid_hws"].to(device)

        # With nothing in the backbone left to train, its graph is dead weight:
        # the detection loss would build one and land on a learning rate of
        # zero. Detection is still computed and still logged, just not
        # differentiated.
        # Inherit the ambient mode when the backbone is trainable rather than
        # forcing grad on, so calling this inside an inference block still means
        # what it says.
        trains_backbone = any(p.requires_grad for p in self.backbone.parameters())
        with nullcontext() if trains_backbone else torch.no_grad():
            backbone_output = self.backbone(
                pixel_values=pixel_values,
                image_grid_hws=image_grid_hws,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                box_token_index=batch["box_token_index"],
                box_token_mask=batch["box_token_mask"],
                labels=batch.get("labels"),
            )

        mask = batch["box_token_mask"].to(device).bool()
        boxes = batch["init_boxes"].to(device)[mask]
        init_contours = batch["init_contours"].to(device)[mask]
        point_mask = batch.get("point_mask")
        if point_mask is not None:
            point_mask = point_mask.to(device)[mask]

        visual_grid, instance_features, tapped = self._contour_inputs(backbone_output)

        images, image_sizes = unpatchify(
            pixel_values, image_grid_hws, self.backbone.patch_size
        )
        features, feature_sizes = self.pixel_decoder(images, image_sizes, visual_grid)

        router_context = self._router_prepare(
            backbone_output,
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            expression_span=batch.get("expression_span"),
            tapped=tapped,
        )
        prediction = self.action_decoder(
            features=features,
            feature_sizes=feature_sizes,
            init_contours=init_contours,
            boxes=boxes,
            batch_index=backbone_output.instance_batch_index,
            instance_features=instance_features,
            point_mask=point_mask,
            distance_head=self.distance_head,
            router=self.router,
            router_context=router_context,
            router_anneal=self._router_anneal(),
        )
        routing = prediction.routing

        distance = None
        if self.distance_head is not None and self.training:
            distance = self.read_distances(
                features,
                feature_sizes,
                boxes,
                backbone_output.instance_batch_index,
            )

        return ContourVLAOutput(
            backbone=backbone_output,
            prediction=prediction,
            boxes=boxes,
            init_contours=init_contours,
            batch_index=backbone_output.instance_batch_index,
            feature_sizes=feature_sizes,
            point_mask=point_mask,
            distance=distance,
            routing=routing,
        )

    def read_distances(
        self,
        features: torch.Tensor,
        feature_sizes: torch.Tensor,
        boxes: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(locations, predicted)`` for the auxiliary loss.

        On a grid over each instance's expanded region rather than on the
        contour: the whole point of this term is to reach the parts of the
        feature map the contour never visits, and a contour that starts as an
        ellipse in a box only ever samples a band around that ellipse.
        """
        settings = self.config.loss
        regions = expand_boxes(boxes, self.config.contour.region_mask_expand)
        locations = region_grid(regions, settings.sdf_grid)
        read = sample_points(features, locations, batch_index, feature_sizes)
        return locations, self.distance_head(read)

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #

    def _router_anneal(self) -> float:
        """Cosine anneal of the near-diagonal prior, shared by every call site
        so the schedule cannot drift between the training forward pass and the
        single-shot ``route`` convenience method."""
        steps = self.config.router.prior_anneal_steps
        if steps <= 0:
            return 1.0
        progress = min(getattr(self, "_router_step", 0) / steps, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _contour_inputs(self, backbone_output):
        """``(visual_grid, instance_features, tapped)`` as the contour branch reads them.

        Detached by default, so a geometric loss - supervised segmentation or
        the RL action likelihood - updates the contour branch while detection
        updates the language model, and neither reaches across. The routed
        text states count too: they are backbone activations like the other
        two, and leaving them attached let the branch write into the LoRA
        weights through the router alone.
        """
        visual_grid = backbone_output.visual_grid
        instance_features = backbone_output.instance_features
        tapped = backbone_output.tapped
        if not self.config.train.segmentation_updates_backbone:
            visual_grid = [grid.detach() for grid in visual_grid]
            instance_features = instance_features.detach()
            tapped = {depth: hidden.detach() for depth, hidden in tapped.items()}
        return visual_grid, instance_features, tapped

    def _router_prepare(self, backbone_output, input_ids, attention_mask,
                         expression_span=None, tapped=None):
        """``(context, valid, keys)`` once per forward pass, or ``None``.

        Image-level and independent of which call or decoder layer will read
        it, so it is computed exactly once and handed to every one of the
        per-call ``LayerRouter.select`` calls the action decoder makes -
        see ``ContourActionChunkDecoder.forward``. ``route`` below is the
        single-shot equivalent for callers that just want one image-level
        selection (inference tooling, ``router diagnostics``).

        Under causal masking the image tokens cannot see the expression that
        follows them, so they carry no language conditioning at any depth; the
        text tokens sit after the image and have attended over it.

        ``expression_span`` narrows this to the referring expression itself.
        Without it the routed context is mostly prompt boilerplate.
        """
        if self.router is None:
            return None
        context, valid = self._router_text(
            backbone_output, input_ids, attention_mask, expression_span, tapped
        )
        keys = self.router.keys(context, valid)
        return context, valid, keys

    def _router_text(self, backbone_output, input_ids, attention_mask,
                     expression_span=None, tapped=None):
        """The parameter-free half of ``_router_prepare``: gathered text states."""
        text_mask = attention_mask.bool() & (
            input_ids != self.backbone.special.image_context
        )
        return self.router.text_context(
            backbone_output.tapped if tapped is None else tapped, text_mask, expression_span
        )

    def route(self, backbone_output, input_ids, attention_mask, batch_index,
              expression_span=None):
        """``(per-slot memory, padding, stats)`` or ``(None, None, None)``.

        Every slot at once, from image-level context alone (no per-instance
        contour/evolution state) - the single-shot equivalent of the per-call
        ``select`` sequence the action decoder actually runs during training
        and generation. Kept for callers that want one routing decision
        without driving the whole decoder, e.g. ``router diagnostics``.
        """
        if self.router is None:
            return None, None, None
        context, valid, keys = self._router_prepare(
            backbone_output, input_ids, attention_mask, expression_span
        )
        return self.router.select(
            context, valid, keys, batch_index, 0, self.router.slots, self._router_anneal()
        )

    def encode_image(self, image):
        """``(pixel_values, image_grid_hws, num_image_tokens)`` for one image.

        Split out of ``predict`` because reinforcement learning replays the same
        image for every rollout in a group, and re-running the vision processor
        eight times for one picture is the kind of waste that only shows up as a
        step time nobody can explain.
        """
        encoded = self.backbone.processor.image_processor(
            images=[image], return_tensors="pt"
        )
        device = self.backbone.device
        grid = np.asarray(encoded["image_grid_hws"]).reshape(-1, 2)
        return (
            encoded["pixel_values"].to(device),
            torch.as_tensor(grid, dtype=torch.int32, device=device),
            int(grid[0, 0] * grid[0, 1]) // (self.backbone.merge_size**2),
        )

    @torch.no_grad()
    def predict(
        self,
        image,
        expression: str,
        max_new_tokens: int | None = None,
        generation_mode: str | None = None,
        contours: bool = True,
        answer_prefix: str = "",
        **generate_kwargs: Any,
    ) -> dict[str, Any]:
        """Detect and segment every instance matching ``expression``.

        Detection runs first because the contour state is defined by the box.
        The generated answer is then replayed through a single teacher-forced
        forward pass: the sampling loop does not expose hidden states, and one
        extra prefill is cheaper than threading a feature tap through it.

        ``contours=False`` stops after detection, so box decoding does not pay
        for the contour replay.

        Decoding settings come from ``config.inference`` rather than from the
        base sampler, whose defaults repeat boxes here; see ``InferenceConfig``.
        Anything passed explicitly wins.
        """
        config = self.config
        device = self.backbone.device
        processor = self.backbone.processor
        tokenizer = self.backbone.tokenizer
        special = self.backbone.special

        settings = config.inference
        sampling = {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "repetition_penalty": settings.repetition_penalty,
        }
        sampling.update(generate_kwargs)
        if max_new_tokens is None:
            max_new_tokens = settings.max_new_tokens
        if generation_mode is None:
            generation_mode = settings.generation_mode

        encoded = processor.image_processor(images=[image], return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device)
        grid = np.asarray(encoded["image_grid_hws"]).reshape(-1, 2)
        image_grid_hws = torch.as_tensor(grid, dtype=torch.int32, device=device)
        num_image_tokens = int(grid[0, 0] * grid[0, 1]) // (self.backbone.merge_size**2)

        prompt, expression_chars = build_prompt(
            expression, num_image_tokens, config.data.prompt_template, with_span=True
        )
        # ``answer_prefix`` continues a partial answer instead of starting one.
        # The decoder takes the prompt as ids, so conditioning on the first half
        # of an earlier answer needs nothing from the generation loop - the
        # prefix simply becomes part of what it is continuing from. Used to
        # branch a decode at the box that went wrong; see
        # prefix diagnostics.
        prompt_ids = tokenizer(
            prompt + answer_prefix, return_tensors="pt", add_special_tokens=False
        )

        continuation = self.backbone.generate_answer(
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
            input_ids=prompt_ids["input_ids"],
            attention_mask=prompt_ids["attention_mask"],
            max_new_tokens=max_new_tokens,
            generation_mode=generation_mode,
            **sampling,
        )
        answer = answer_prefix + continuation

        boxes_np = parse_boxes(answer)
        result: dict[str, Any] = {"answer": answer, "boxes": boxes_np, "contours": None}
        if boxes_np.shape[0] == 0 or not contours:
            return result

        result.update(
            self.contours_for_answer(
                pixel_values, image_grid_hws, prompt, answer, expression_chars
            )
        )
        return result

    def contours_for_answer(
        self, pixel_values, image_grid_hws, prompt: str, answer: str,
        expression_chars: tuple[int, int] | None = None,
        *,
        return_tensors: bool = False,
        action_std: float = 0.0,
        action_latents: torch.Tensor | None = None,
        init_polygons: list | None = None,
        backbone_grad: bool = True,
        deterministic_too: bool = False,
    ) -> dict[str, Any]:
        """Evolve the boxes written in ``answer`` into contours.

        ``return_tensors`` retains the graph for joint RL, including grounding
        hidden states for no-target answers. Stochastic actions are opt-in;
        the public inference result remains deterministic and NumPy-based.

        ``backbone_grad=False`` runs the backbone without a graph, for an RL
        replay that only needs the contour branch's likelihood.
        ``deterministic_too`` additionally evolves the action means from the
        same backbone pass and returns them as ``mean_prediction``: the contour
        inference would draw, which RL prices grounding with and uses as the
        per-instance baseline for the sampled actions.

        This is the teacher-forced half of ``predict``, which used to be inlined
        there. It is separate because reinforcement learning has to price
        answers it did not just generate: every rollout in a group needs its
        boxes turned into the masks the reward is computed from, and the reward
        must be the mask this system would actually produce rather than a
        stand-in.

        Returns an empty mapping when the answer contains no box, which is the
        refusal case and not an error.
        """
        device = self.backbone.device
        tokenizer, special, config = self.backbone.tokenizer, self.backbone.special, self.config

        prompt_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False,
            return_offsets_mapping=expression_chars is not None,
        )
        answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"]
        input_ids = torch.cat([prompt_ids["input_ids"].to(device), answer_ids.to(device)], dim=1)
        attention_mask = torch.ones_like(input_ids)

        answer_spans = find_box_token_spans(answer_ids[0].tolist(), special)
        spans = answer_spans + prompt_ids["input_ids"].shape[1]
        if spans.shape[0] == 0 and not return_tensors:
            return {}
        boxes_np = decode_box_spans(input_ids[0].tolist(), spans, special)

        box_token_index = torch.as_tensor(spans, dtype=torch.long, device=device)[None]
        box_token_mask = torch.ones(
            (1, spans.shape[0]), dtype=torch.bool, device=device
        )

        with nullcontext() if backbone_grad else torch.no_grad():
            backbone_output = self.backbone(
                pixel_values=pixel_values,
                image_grid_hws=image_grid_hws,
                input_ids=input_ids,
                attention_mask=attention_mask,
                box_token_index=box_token_index,
                box_token_mask=box_token_mask,
                labels=None,
            )

        if spans.shape[0] == 0:
            if action_latents is not None and action_latents.shape[0] != 0:
                raise ValueError("nonempty action replay for an answer without valid boxes")
            return {"boxes": boxes_np, "prediction": None, "backbone": backbone_output,
                    "point_mask": None, "point_counts": np.zeros(0, dtype=np.int64),
                    "box_spans": []}

        # The budget comes from the boxes the detector just produced, which is
        # the same rule training applied to the jittered ground truth.
        settings = config.contour
        if settings.dynamic_points:
            counts = point_budget(
                boxes_np,
                settings.num_points,
                settings.min_points,
                settings.point_spacing_px,
                settings.point_reference_side,
                settings.point_quantum,
            )
        else:
            counts = np.full((boxes_np.shape[0],), settings.num_points, dtype=np.int64)
        contour_array, mask_array = boxes_to_padded_contours(
            boxes_np, counts, settings.num_points, settings.init_mode
        )
        if init_polygons is not None:
            for index, polygon in enumerate(init_polygons):
                if polygon is None or index >= contour_array.shape[0]:
                    continue
                count = int(counts[index])
                polygon = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if polygon.shape[0] == count:
                    contour_array[index, :count] = polygon
                else:
                    contour_array[index, :count] = resample_closed_polygon(polygon, count)
        init_contours = torch.as_tensor(contour_array, device=device)
        point_mask = torch.as_tensor(mask_array, device=device)
        boxes = torch.as_tensor(boxes_np, device=device, dtype=init_contours.dtype)

        visual_grid, instance_features, tapped = self._contour_inputs(backbone_output)
        images, image_sizes = unpatchify(
            pixel_values, image_grid_hws, self.backbone.patch_size
        )
        # Must match training: routing a different set of positions at
        # inference than in training is a silent train/test mismatch.
        expression_span = None
        if expression_chars is not None:
            span = token_span_for_chars(
                prompt_ids["offset_mapping"][0].tolist(), expression_chars
            )
            expression_span = torch.tensor([span], device=device)
        router_text = None
        if self.router is not None:
            router_text = self._router_text(
                backbone_output, input_ids, attention_mask, expression_span, tapped=tapped
            )
        context = {
            "images": images, "image_sizes": image_sizes, "visual_grid": visual_grid,
            "instance_features": instance_features,
            "batch_index": backbone_output.instance_batch_index,
            "router_text": router_text, "init_contours": init_contours, "boxes": boxes,
            "point_mask": point_mask,
        }
        features, feature_sizes, router_context = self._contour_features(context)
        prediction = self._evolve(context, features, feature_sizes, router_context,
                                  action_std, action_latents)

        if return_tensors:
            result = {"boxes": boxes, "prediction": prediction, "backbone": backbone_output,
                      "point_mask": point_mask, "point_counts": counts,
                      "box_spans": [(int(s[0]), int(s[-1]) + 1) for s in answer_spans]}
            if deterministic_too:
                result["mean_prediction"] = (
                    self._evolve(context, features, feature_sizes, router_context, 0.0, None)
                    if action_std > 0 else prediction
                )
            if not self.config.train.segmentation_updates_backbone:
                # Everything the contour branch read from the backbone, already
                # detached, so an action replay can skip the backbone entirely.
                result["context"] = context
            return result
        return {
            "boxes": boxes_np,
            "contours": prediction.final.detach().float().clamp(0.0, 1.0).cpu().numpy(),
            "init_contours": init_contours.detach().float().cpu().numpy(),
            # Consumers must slice each contour to its own length; the padded
            # tail holds a repeat of the initial point, which would close the
            # polygon through the wrong place if drawn.
            "point_counts": counts,
        }

    def _contour_features(self, context):
        """Pixel features and router keys: the trainable part before the decoder."""
        features, feature_sizes = self.pixel_decoder(
            context["images"], context["image_sizes"], context["visual_grid"]
        )
        router_context = None
        if self.router is not None and context["router_text"] is not None:
            text, valid = context["router_text"]
            router_context = (text, valid, self.router.keys(text, valid))
        return features, feature_sizes, router_context

    def _evolve(self, context, features, feature_sizes, router_context, action_std, action_latents):
        return self.action_decoder(
            features=features,
            feature_sizes=feature_sizes,
            init_contours=context["init_contours"],
            boxes=context["boxes"],
            batch_index=context["batch_index"],
            instance_features=context["instance_features"],
            point_mask=context["point_mask"],
            distance_head=self.distance_head,
            router=self.router,
            router_context=router_context,
            router_anneal=self._router_anneal(),
            action_std=action_std,
            action_latents=action_latents,
        )

    def evolve_from_context(self, context, action_std: float = 0.0, action_latents=None):
        """Re-run the contour branch from a ``contours_for_answer`` context.

        The context holds detached backbone read-outs, so this rebuilds exactly
        the graph the branch's own parameters need - pixel decoder, router keys,
        action decoder - without another pass through the 3B backbone.
        """
        features, feature_sizes, router_context = self._contour_features(context)
        return self._evolve(context, features, feature_sizes, router_context,
                            action_std, action_latents)


def unpatchify(
    pixel_values: torch.Tensor, image_grid_hws: torch.Tensor, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild padded image tensors from the backbone's patch sequence.

    Reusing the patches the vision tower already consumes keeps the pixel
    decoder pixel-perfectly aligned with the language model's view of the image
    and avoids decoding every image twice.
    """
    grids = image_grid_hws.to(torch.int64).reshape(-1, 2)
    heights = (grids[:, 0] * patch_size).tolist()
    widths = (grids[:, 1] * patch_size).tolist()
    max_height, max_width = max(heights), max(widths)

    images = pixel_values.new_zeros((grids.shape[0], 3, max_height, max_width))
    offset = 0
    for index, (grid_h, grid_w) in enumerate(grids.tolist()):
        count = grid_h * grid_w
        patches = pixel_values[offset : offset + count]
        offset += count
        image = patches.reshape(grid_h, grid_w, 3, patch_size, patch_size)
        image = image.permute(2, 0, 3, 1, 4).reshape(
            3, grid_h * patch_size, grid_w * patch_size
        )
        images[index, :, : grid_h * patch_size, : grid_w * patch_size] = image

    sizes = torch.stack(
        [
            torch.as_tensor(heights, device=pixel_values.device),
            torch.as_tensor(widths, device=pixel_values.device),
        ],
        dim=1,
    )
    return images, sizes
