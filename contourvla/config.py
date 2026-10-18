"""Configuration dataclasses for ContourVLA."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Mapping

# The repo root, two levels up from this file (contourvla/config.py). Relative
# weight and data paths in the configs resolve against this - see
# ContourVLAConfig.resolve_paths.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class BackboneConfig:
    """Vision-language interaction module."""

    checkpoint_path: str = "ckpts/contourvla.pt"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    # Which decoder layer each contour input is tapped from; -1 is the final
    # (post-norm) layer. The two differ because they carry different things.
    # The box descriptor sits after the expression and is where the referring
    # decision resolves into coordinates, so it wants the last layer. The image
    # tokens sit *before* the expression, so under the causal mask they never
    # see it at any depth. The grid therefore reads an earlier layer.
    instance_layer: int = -1
    grid_layer: int = 18
    # Number of visual patches (pre 2x2 merge) allowed per image. 4096 is the
    # released default; lowering it trades boundary detail for speed/memory.
    in_token_limit: int = 2560
    max_text_len: int = 1024

    freeze_vision: bool = True
    freeze_llm: bool = False
    freeze_projector: bool = False
    llm_lora_rank: int = 32
    llm_lora_alpha: int = 64
    llm_lora_dropout: float = 0.05
    gradient_checkpointing: bool = True


@dataclass
class PixelDecoderConfig:
    """Builds the high-resolution feature map the contour points move on."""

    hidden_dim: int = 256
    # Output stride relative to the processed input image. 2 rather than 4, so
    # the feature map is fine enough for the contour to be told where the
    # boundary is. Bilinear sampling interpolates between features but cannot
    # invent information that is not in them.
    output_stride: int = 2
    # 'channel' normalises over channels at each position. 'group' pools over the
    # whole spatial map and so mixes in the right/bottom padding that batching
    # adds, which makes an image's features depend on its batch-mates. Inference
    # runs one image at a time with no padding.
    norm: str = "channel"
    # A pyramid over the stem. The stem has resolution and a short range; the
    # language grid has range and a coarse stride. 0 restores the flat stem.
    pyramid_levels: int = 0
    pyramid_depth: int = 2  # conv blocks per level, each way
    stem_dims: tuple[int, ...] = (64, 128, 128)
    # Longest side the CNN stem operates on. The stem is the only part of the
    # network that sees full-resolution pixels, so this caps its cost.
    max_stem_side: int = 896
    num_fusion_blocks: int = 2
    use_llm_features: bool = True
    dropout: float = 0.0


@dataclass
class ActionDecoderConfig:
    """Contour action-chunk decoder (the 'action expert')."""

    num_points: int = 64
    hidden_dim: int = 256
    nheads: int = 8
    # Depth of the contour transformer.
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    # Length of one predicted action chunk: each contour point receives
    # `chunk_size` successive displacements in a single forward pass.
    chunk_size: int = 4
    # How many times the chunk decoder is re-invoked on re-sampled features
    # (open-loop chunk execution followed by re-observation).
    num_calls: int = 2
    share_call_weights: bool = True
    # Per-step displacement cap, in units of the box width/height. The total
    # budget is `num_calls * chunk_size * max_step`. Deep supervision also asks
    # every call to reach the ground truth on its last step, so one call's
    # budget has to be able to get there. Excess headroom keeps tanh in its
    # narrow linear band.
    max_step: float = 0.2
    # ROIAlign grid used as the decoder memory.
    roi_output_size: int = 7
    roi_expand: float = 1.25
    instance_dim: int = 2048  # LLM hidden size; overwritten from the backbone.
    # Put the instance tokens of the other objects in the same image into each
    # instance's memory, so neighbouring contours can avoid claiming the same
    # pixels. Only the summary tokens are shared, not the contour points.
    cross_instance: bool = True
    # Harmonics in the arc-position encoding. A contour is a closed curve, so a
    # point's place on it is an angle, not an index: sin/cos of 2*pi*k*t are
    # exactly periodic, which a learned per-index table is not - there, the last
    # point and the first are neighbours in geometry and strangers in the
    # embedding. Being a function of t = i/count rather than of i is also what
    # lets one set of weights serve every point budget. Nyquist would ask for
    # count/2 harmonics to separate adjacent points on the finest contour, but
    # the queries also carry their own coordinates and sampled features, so the
    # encoding only has to place them smoothly.
    # Below Nyquist for the finest contour. Higher harmonics alias on shorter
    # contours, and the queries already carry coordinates and sampled features.
    arc_harmonics: int = 32
    # Buckets for the relative arc distance between two points, log-spaced per
    # direction. Attention over a contour should depend on how far apart two
    # points are along it, and "far" has to mean the same thing at every budget:
    # with absolute encodings alone, a model trained where neighbours sit 1/64 of
    # a turn apart meets neighbours 1/128 apart and silently changes behaviour.
    arc_buckets: int = 32
    # Samples taken along each point's outward normal, and how far they reach in
    # units of the box. A point otherwise reads one feature at its own location,
    # while it may be asked to move a large fraction of the box. 0 restores the
    # single sample. A wider span needs more samples or it starts skipping.
    normal_samples: int = 11
    normal_span: float = 0.25  # half-range, so +/-0.25 box units

    # Append the auxiliary head's predicted distance to each profile sample, so
    # the query carries a distance reading along the normal whose zero crossing
    # is where the point should land, instead of a feature vector the decoder has
    # to infer that from. Off by default and deliberately: the head is itself
    # learned, and feeding a channel the decoder comes to rely on is only worth
    # it once that channel is accurate. `sdf_r2` is logged every step so the
    # decision is made on a number - see LossConfig.sdf_weight.
    sdf_feedback: bool = False
    # Keep the contour loss out of the head. The channel is useful precisely
    # because it is a true distance field; letting the point loss shape it too
    # would drift it towards whatever suits the decoder, which is the property
    # being paid for.
    sdf_detach: bool = True
    # Set from RouterConfig.enabled by ContourVLA. Only then are the routed
    # cross-attentions built: unbuilt beats built-and-never-called, which is a
    # block of parameters sitting in the optimiser receiving no gradient.
    routed_context: bool = False


@dataclass
class ContourConfig:
    """Contour geometry, including how many points each object gets.

    ``num_points`` is the allocation - every instance gets a tensor this wide -
    while the number actually evolved varies per object between ``min_points``
    and ``num_points``. The unused tail is masked out of attention and of the
    loss, so a small object costs the same tensor but not the same supervision.

    The rule is constant arc spacing rather than a threshold on area, because
    that is what the artefact it fixes depends on. A 64-point contour on an
    object covering a large part of the frame has long straight chords, which
    reads as a polygon; on a small object the same count sits too close
    together. Spacing captures both ends with one number, and area does not.
    """

    num_points: int = 128  # allocated per instance, and the ceiling of the rule
    min_points: int = 64  # floor, for objects too small to use more
    dynamic_points: bool = True  # off pins every instance at num_points
    point_spacing_px: float = 3.0  # target chord length on screen
    point_reference_side: int = 640  # raster the spacing is quoted against
    point_quantum: int = 8  # counts are rounded to a multiple of this
    init_mode: str = "ellipse"  # ellipse | rectangle | octagon
    # Post-processing of the predicted contour, applied when it is rasterised or
    # drawn and never during training. ``smooth_harmonics`` truncates the
    # outline's spectrum, which is what removes the zigzag and with it most small
    # self-crossings; ``render_points`` evaluates the truncated series more
    # densely, which is what removes the faceting that no point budget under a
    # few hundred can. 0 disables either half. Resampling without the
    # band-limit interpolates the zigzag and can add crossings rather than
    # remove them.
    smooth_harmonics: int = 16
    render_points: int = 256
    # Grid the region loss is measured on, and how far around the box it
    # reaches. Geometry rather than loss weighting, because the dataset has to
    # crop the ground truth to exactly the region the rasteriser will use - the
    # two lining up is the whole correctness condition. 0 skips the crop.
    # Resolution is paired with the sigmoid temperature: a sharper band needs
    # more cells if any of them are to carry a gradient.
    region_mask_resolution: int = 128
    region_mask_expand: float = 1.6  # room for a contour that overshoots
    # How the ground-truth contour is put in one-to-one correspondence with the
    # initial contour: uniform arc-length sampling plus the cyclic shift and
    # orientation that minimise the total travel distance.
    match_mode: str = "arclength_aligned"  # arclength_aligned | polar
    gt_source: str = "polygon"  # polygon | mask
    mask_raster_side: int = 512
    # Random box jitter applied to the initial contour during training so the
    # decoder is robust to imperfect detections at test time. Training starts
    # from ground-truth boxes and inference runs on predicted ones, and jitter
    # is the only thing bridging that gap. It is ramped rather than fixed: a
    # sloppy initialisation is hard to learn from before the policy can do
    # anything, and too clean an one leaves the decoder unprepared for the
    # detector it will actually be paired with.
    box_jitter: float = 0.05
    box_jitter_final: float = 0.15
    box_jitter_ramp_steps: int = 10000


@dataclass
class LossConfig:
    detection_weight: float = 1.0
    segmentation_weight: float = 2.0
    # Value substituted for the segmentation loss when an image yields no usable
    # instance (no detection / no valid polygon). Keeps the total loss on a
    # stable scale instead of silently dropping to zero.
    segmentation_fallback: float = 0.25
    # Displacements are supervised in box-normalised units, so the loss is
    # scale invariant and the fallback constant stays comparable across images.
    normalize_by_box: bool = True
    smooth_l1_beta: float = 0.05
    # Weight ramp across chunk steps: later steps matter more.
    step_weight_power: float = 1.0
    # Two regularisers aimed at contours that cross themselves. Neither forbids
    # it - a differentiable test for segment intersection is O(P^2) and not
    # smooth - but both remove the configurations it grows out of. Curvature
    # penalises a point sitting far from the midpoint of its neighbours, which
    # is what a spike is; spacing penalises the spread of chord lengths, because
    # points bunching together is what lets neighbours swap order. Both are
    # measured in box-normalised units and averaged over an instance's real
    # points, so they do not depend on its point budget.
    smoothness_weight: float = 0.02
    spacing_weight: float = 0.05
    # Region loss. The point term cannot see area, so this compares a raster of
    # the contour with the true mask, both cropped to the instance's expanded
    # box (see contourvla/contour/raster.py). 0 disables it. Kept near parity
    # with the point term: soft IoU cannot see a point slide along the boundary,
    # and letting it dominate frees the arc-length correspondence to drift.
    region_weight: float = 0.35
    # Sigmoid width, in units of the region's own size. It decides whether this
    # term optimises the shape or a blur of it. A wide band makes the soft mask
    # a poor stand-in for the hard one.
    soft_iou_temperature: float = 0.0025
    # Points whose target sits far from where they start are the ones that never
    # arrive - a protrusion is a handful of points that must travel several times
    # further than the rest, and a mean over points buries them. This tilts the
    # point loss towards them: weight = 1 + travel_weight * (travel / mean
    # travel). 0 restores the plain mean.
    travel_weight: float = 1.0

    # Auxiliary dense supervision on the pixel decoder's features: a small head
    # reads a feature and must say how far the boundary is.
    #
    # The point loss cannot ask this. It reaches the features only through the
    # action decoder, and only at the locations a contour samples, in a band
    # around an ellipse inscribed in the jittered box. Most of the map is never
    # asked anything. What it asks for is a displacement, which the
    # decoder can produce from the region grid and the instance token without the
    # feature encoding a distance at all. The head regresses that distance.
    # 0 disables the term. An auxiliary should shape the representation rather
    # than become the objective, so the weight stays below the segmentation term.
    sdf_weight: float = 0.0
    sdf_grid: int = 16  # sample points per side, per instance
    # Truncation, in units of the box. Matches ActionDecoderConfig.normal_span:
    # beyond the profile's reach the distance can be neither read nor used, and
    # untruncated far values would dominate an L1.
    sdf_truncation: float = 0.25
    sdf_hidden: int = 64


@dataclass
class RouterConfig:
    """Dynamic routing of language-model layers into the action decoder.

    After MoRoute (arXiv 2607.29545): rather than hand-picking which layers the
    contour branch reads - the defaults are layer 18 for the grid and the last for
    the box descriptor - each decoder layer learns which one it wants.

    Two departures. The routed values are the *text*
    positions, not the whole sequence: under causal masking an image token never
    sees the expression that follows it, so no layer of the image grid is
    language-conditioned, while text tokens sit after the image and have
    attended over it. And routing is per (decoder layer, call) rather than per
    layer, because the two calls are separate stages of one evolution, which is
    the same argument MoRoute makes for routing by denoising step.
    """

    enabled: bool = False
    # Every ``stride``-th language-model layer is a candidate. All 36 would cost
    # roughly 600 MB of activations at batch 4.
    stride: int = 4
    key_dim: int = 64
    # Text tokens kept, counted from the end of the prompt: a referring
    # expression is short and sits last, so truncating from the front drops the
    # system prompt rather than the description.
    max_tokens: int = 32
    prior_scale: float = 4.0
    # Steps over which the near-diagonal prior is annealed away. Without it
    # MoRoute reports routing either unstable or collapsed early on.
    prior_anneal_steps: int = 1000
    diversity_weight: float = 0.01
    confidence_weight: float = 0.01
    monotonic_weight: float = 0.001
    # Fold per-instance contour geometry and per-call evolution progress (how
    # far the previous call actually moved the contour) into the query before
    # it meets the keys, so instances sharing an image - or one instance across
    # its two calls - are no longer forced to make the same pick. The offset
    # this adds starts at exactly zero (its last layer is zero-initialised),
    # so turning it on changes nothing until training moves it; see
    # models/router.py. False reproduces the original, image-level-only router.
    state_conditioning: bool = True


@dataclass
class RLConfig:
    """Grounding-only or joint grounding/continuous-action post-training."""

    joint_training: bool = False  # enabled explicitly by configs/rl.yaml
    action_std: float = 0.05  # fixed exploration std before tanh; inference stays deterministic
    action_loss_weight: float = 1.0
    group: int = 8
    temperature: float = 0.7
    temperature_jitter: float = 0.05  # per-rollout uniform radius; 0 disables jitter
    top_p: float = 0.9
    epsilon: float = 0.1  # entropic temperature of the transport plan
    dustbin: float = 0.25  # what an unmatched prediction is worth
    credit_blend: float = 0.5  # share of the advantage that is per-object
    clip: float = 0.2
    kl_weight: float = 0.04
    # Below this group standard deviation the advantage is zeroed. GRPO divides
    # by the spread, so a group whose rollouts all scored alike does not produce
    # a small update. It produces a unit-scale one built from rasterisation
    # jitter. Groups below this spread are zeroed.
    advantage_floor: float = 0.02
    # The same floor for the joint path's action advantages, which are the
    # sampled contour's credit minus the deterministic contour's for the same
    # box. They live on a finer scale than grounding rewards, so they get their
    # own threshold instead of dragging the grounding floor down with them.
    action_advantage_floor: float = 0.001
    # KL anchor for the answer tokens. "lora_off" is the base VLM with the LoRA
    # gain at zero - the projector stays trained, so it is not even a fixed
    # model - and every RL stage started from a fine-tuned checkpoint is pulled
    # back toward it. "initial" snapshots every trainable backbone tensor when
    # the stage starts, the usual reference for continuing RL.
    reference: str = "lora_off"
    # Carry the backbone's AdamW moments over from a train.init_from checkpoint
    # whose optimizer covers exactly those tensors (a previous RL stage), so the
    # continued run does not restart them at zero.
    inherit_optimizer: bool = False

    # Share of RL steps spent on no-target images. They are a small fraction of
    # the data, so the natural rate under-trains refusal. 0 keeps that rate.
    # The share is changed here rather than by raising negative_alpha, so the
    # reward definition stays fixed.
    negative_share: float = 0.25

    force_refusal: bool = True
    negative_alpha: float = 0.5


@dataclass
class DataConfig:
    dataset: str = "grefcoco"  # grefcoco | refcoco | jsonl | synthetic
    root: str = "data"
    ref_file: str = ""  # defaults resolved per dataset
    instances_file: str = ""
    image_root: str = ""
    split: str = "train"
    split_by: str = "unc"
    max_instances: int = 24
    # gRefCOCO contains "no-target" expressions; keeping them teaches the
    # detector to emit `<box>none</box>`.
    keep_no_target: bool = True
    min_box_size: float = 0.005  # normalised, drops degenerate annotations
    prompt_template: str = (
        "Locate all the instances that match the following description: {expr}."
    )
    num_workers: int = 8
    horizontal_flip: float = 0.0  # unsafe for left/right referring expressions
    # The answer echoes the referring expression back inside ``<ref>...</ref>``
    # before the boxes. Supervising that copy is what most of the detection
    # loss was actually spent on; see ``ContourReferringDataset._drop_ref_text``.
    supervise_ref_text: bool = False
    # Flattened annotations are cached here so every rank does not re-parse the
    # 180 MB of JSON. Empty disables caching.
    cache_dir: str = ""
    # Split held out for the periodic visualisation and evaluation.
    val_split: str = "val"


@dataclass
class InferenceConfig:
    """How boxes are decoded.

    ``slow`` is pure autoregression and the default. ``hybrid`` decodes a whole
    box in one parallel block and then decides whether another follows. Under
    greedy that decision repeats until ``max_new_tokens`` runs out. The sampling
    fields stay because ``slow`` accepts them. Under greedy they are inert.
    """

    generation_mode: str = "slow"  # slow (pure autoregression) | hybrid
    max_new_tokens: int = 512
    temperature: float = 0.0  # 0 selects greedy inside the base sampler
    top_p: float | None = None
    repetition_penalty: float = 1.0
    seed: int = 1234  # only matters if temperature is raised


@dataclass
class VisualizationConfig:
    """Periodic segmentation overlays, so training can be watched rather than
    inferred from a loss curve."""

    every: int = 500  # steps between renders; 0 disables
    num_samples: int = 8
    # Render from the held-out split with real generation, which is what the
    # model is actually judged on. Teacher-forced training samples are cheaper
    # and show the contour branch in isolation.
    source: str = "both"  # train | val | both
    max_side: int = 640
    save_to_disk: bool = True
    # Also draw the box contour the evolution started from.
    draw_initial: bool = True


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "contourvla"
    entity: str = ""
    run_name: str = ""
    group: str = ""
    tags: tuple[str, ...] = ()
    mode: str = "disabled"  # online | offline | disabled
    # these before `wandb.init` when they are not already set.
    no_proxy: str = "localhost,127.0.0.1"
    activate_proxy: bool = False
    api_key: str = ""  # falls back to $WANDB_API_KEY


@dataclass
class TrainConfig:
    output_dir: str = "outputs/contourvla"
    batch_size: int = 2
    grad_accum: int = 8
    max_steps: int = 40000
    warmup_steps: int = 1000
    lr: float = 1e-4
    backbone_lr: float = 2e-5
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    seed: int = 0
    # Route each loss to its own parameters: detection to the language model,
    # segmentation to the contour branch. When this is false the contour loss
    # does not flow into the backbone.
    segmentation_updates_backbone: bool = False
    # Stop updating the backbone at this step; 0 trains it for the whole run.
    # Implemented by taking the group's learning rate to zero rather than by
    # clearing requires_grad, which would leave DDP expecting gradients that
    # never arrive.
    freeze_backbone_after: int = 0
    log_every: int = 20
    save_every: int = 2000
    # Written to ``best.pt`` whenever validation generation improves on it.
    # Empty disables. A validation metric, not the training loss.
    best_metric: str = "mask_iou"
    eval_every: int = 0
    eval_samples: int = 200
    amp_dtype: str = "bfloat16"
    # Continue an interrupted run: restores weights, step counter and optimizer
    # state, so the schedules pick up where they stopped.
    resume: str = ""
    # Start a *new* run from someone else's weights. Step counter and optimizer
    # begin at zero, so the cosine anneals over this run's max_steps rather than
    # resuming halfway up a schedule that belonged to a different horizon.
    init_from: str = ""
    # Which parameters ``init_from`` brings across, by prefix. Empty takes all.
    init_keep: str = ""
    # Substrings removed after init_keep selects, for submodules that do not sit
    # under a prefix of their own - the routed cross-attention is inside
    # action_decoder.blocks.*, so no allowlist can keep the decoder and drop it.
    init_drop: str = ""


@dataclass
class ContourVLAConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    pixel_decoder: PixelDecoderConfig = field(default_factory=PixelDecoderConfig)
    action_decoder: ActionDecoderConfig = field(default_factory=ActionDecoderConfig)
    contour: ContourConfig = field(default_factory=ContourConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        # `num_points` lives in two places for readability; keep them in sync.
        self.action_decoder.num_points = self.contour.num_points

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ContourVLAConfig":
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            value = mapping.get(f.name)
            if value is None:
                continue
            if is_dataclass(f.type) or isinstance(value, Mapping):
                sub_cls = cls.__dataclass_fields__[f.name].default_factory  # type: ignore[misc]
                kwargs[f.name] = _build(sub_cls(), value)
            else:
                kwargs[f.name] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str) -> "ContourVLAConfig":
        config = cls.from_mapping(_read_with_extends(path, set()))
        config.resolve_paths()
        return config

    def resolve_paths(self) -> None:
        """Make the checkpoint and data paths absolute, against the repo root.

        The configs name these relatively - ``ckpts/contourvla.pt``, ``data`` - so that
        a checkout is self-contained and nothing has to be edited to reproduce a
        run elsewhere. They resolve against the repo rather than the process's
        working directory because a relative path that follows the caller around
        means something different once a launcher changes directory. An absolute
        path in the yaml is left exactly as written.
        """
        for holder, field_name in (
            (self.backbone, "checkpoint_path"),
            (self.data, "root"),
            (self.data, "cache_dir"),
            (self.data, "ref_file"),
            (self.data, "instances_file"),
            (self.data, "image_root"),
        ):
            value = getattr(holder, field_name)
            if value and not os.path.isabs(value):
                setattr(holder, field_name, os.path.join(REPO_ROOT, value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)


def _read(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith((".yaml", ".yml")):
            import yaml

            payload = yaml.safe_load(handle)
        else:
            payload = json.load(handle)
    return payload or {}


def _read_with_extends(path: str, seen: set[str]) -> dict[str, Any]:
    """Load a config, resolving one optional ``extends: <path>``.

    Two training stages share almost every setting, and a second full copy of a
    250-line file is a copy that drifts: the halves that matter get edited in one
    of them and the difference between the stages stops being what the diff says
    it is. A stage file states only what it changes.

    The parent path is relative to the child, and a cycle is an error rather than
    a hang.
    """
    resolved = os.path.abspath(path)
    if resolved in seen:
        raise ValueError(f"config extends itself: {' -> '.join(sorted(seen))} -> {path}")
    seen.add(resolved)

    payload = _read(path)
    parent = payload.pop("extends", None)
    if parent is None:
        return payload

    if not os.path.isabs(parent):
        parent = os.path.join(os.path.dirname(resolved), parent)
    return _merge(_read_with_extends(parent, seen), payload)


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Section-wise, so a stage that changes one key in ``train`` keeps the rest."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


# Keys that existed in configs written by earlier runs and have since been
# removed from the dataclasses. Every run directory holds a config.json, and
# reloading one is the documented way to evaluate that checkpoint
# (scripts/evaluate.py says so when it refuses a mismatched architecture), so a
# retired key has to be ignored rather than rejected - otherwise deleting a field
# silently orphans every checkpoint trained before it went away. An unknown key
# that is *not* on this list is still an error, because that is a typo.
RETIRED_KEYS = frozenset({
    # Stage-2 self-forcing: declared, never implemented, removed in the config
    # consolidation. Present in every config.json written before then.
    "self_forcing_ratio",
    "self_forcing_match_iou",
})


def _build(instance: Any, mapping: Mapping[str, Any]) -> Any:
    for key, value in mapping.items():
        if not hasattr(instance, key):
            if key in RETIRED_KEYS:
                continue
            raise KeyError(f"unknown config key {type(instance).__name__}.{key}")
        setattr(instance, key, value)
    return instance
