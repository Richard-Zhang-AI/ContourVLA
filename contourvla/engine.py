"""Training and evaluation loops."""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ContourVLAConfig
from .contour.geometry import (
    box_iou,
    contour_to_mask,
    match_boxes_one_to_one,
    segmentation_to_contour,
    smooth_contour,
)
from .data.collate import collate_contour_batch, flatten_instances
from .losses import ContourVLALoss
from .models.contourvla import ContourVLA
from .models.router import routing_losses
from .visualize import (
    pad_to_common_size,
    save_panels,
    visualize_generated,
    visualize_teacher_forced,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def cosine_schedule(step: int, warmup: int, total: int, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def build_dataloader(
    dataset, config: ContourVLAConfig, pad_token_id: int, shuffle: bool, distributed: bool
) -> DataLoader:
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.data.num_workers > 0,
        collate_fn=lambda samples: collate_contour_batch(samples, pad_token_id),
    )


class Trainer:
    def __init__(
        self,
        model: ContourVLA,
        config: ContourVLAConfig,
        train_loader: DataLoader,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        val_samples=None,
        logger=None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.val_samples = list(val_samples or [])
        self.logger = logger

        self.core: ContourVLA = model.module if hasattr(model, "module") else model
        self.criterion = ContourVLALoss(config.loss)

        groups = self.core.parameter_groups(
            lr=config.train.lr,
            backbone_lr=config.train.backbone_lr,
            weight_decay=config.train.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.step = 0
        self.best_value: float | None = None
        self.best_step = 0
        self._announced_freeze = False
        self._last_batch: dict[str, Any] | None = None
        self._last_output = None

    def _advance_data_schedule(self) -> None:
        """Publish the step count to the dataset (and its worker processes)."""
        dataset = getattr(self.train_loader, "dataset", None)
        if hasattr(dataset, "set_step"):
            dataset.set_step(self.step)

    def _current_jitter(self) -> float:
        dataset = getattr(self.train_loader, "dataset", None)
        return float(dataset.current_jitter()) if hasattr(dataset, "current_jitter") else 0.0

    def _set_lr(self) -> float:
        config = self.config.train
        scale = cosine_schedule(self.step, config.warmup_steps, config.max_steps)
        # The backbone is the first group (see ContourVLA.parameter_groups) and
        # the only one the freeze applies to. Zeroing its rate rather than
        # clearing requires_grad keeps the gradients flowing, which DDP is built
        # around; AdamW's decoupled decay also scales with the rate, so zero
        # really does mean frozen rather than "decaying towards zero".
        frozen = bool(config.freeze_backbone_after) and self.step >= config.freeze_backbone_after
        for index, (group, base) in enumerate(zip(self.optimizer.param_groups, self.base_lrs)):
            backbone_group = index == 0 and len(self.optimizer.param_groups) > 1
            group["lr"] = 0.0 if (frozen and backbone_group) else base * scale
        if frozen and not self._announced_freeze:
            self._announced_freeze = True
        return self.base_lrs[-1] * scale

    def compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        output = self.model(batch)
        device = self.device
        targets = flatten_instances(batch, "target_contours", device)
        valid = flatten_instances(batch, "contour_valid", device)
        loss, metrics = self.criterion(
            detection_loss=output.backbone.detection_loss,
            prediction=output.prediction,
            target_contours=targets,
            boxes=output.boxes,
            valid=valid,
            contour_modules=self.core.contour_modules,
            point_mask=output.point_mask,
            region_boxes=flatten_instances(batch, "region_boxes", device)
            if "region_boxes" in batch else None,
            region_masks=flatten_instances(batch, "region_masks", device)
            if "region_masks" in batch else None,
            distance=output.distance,
        )
        if output.routing is not None:
            settings = self.config.router
            # Keeps the router from collapsing onto a couple of layers, from
            # staying so soft that the straight-through estimator is fighting a
            # near-uniform distribution, and from ignoring depth order.
            loss = loss + routing_losses(
                output.routing["probabilities"],
                settings.diversity_weight,
                settings.confidence_weight,
                settings.monotonic_weight,
            )
            selection = output.routing["selection"].float()  # [B, slots]
            metrics["route_layer"] = float(selection.mean())
            metrics["route_spread"] = float(selection.std(unbiased=False))
            # Share of slots that pick different layers for different images in
            # the batch. route_spread cannot see this - it stayed at exactly
            # 2.5495 for 6000 steps while every image was routed identically,
            # because a fixed spread-out assignment and an adaptive one look the
            # same to it. This is the number that says whether routing routes.
            if selection.shape[0] > 1:
                metrics["route_adapt"] = float(
                    (selection.std(dim=0, unbiased=False) > 0).float().mean()
                )
            self.core._router_step = self.step
        self._last_batch, self._last_output = batch, output
        return loss, metrics

    def train(self, log_fn=print) -> None:
        config = self.config.train
        self.model.train()
        accumulated: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        started = time.time()
        iterator = _infinite(self.train_loader)

        while self.step < config.max_steps:
            was_announced = self._announced_freeze
            learning_rate = self._set_lr()
            if self._announced_freeze and not was_announced:
                log_fn(
                    f"[freeze] step {self.step}: backbone learning rate is now 0; "
                    f"only the contour branch keeps training"
                )
            self._advance_data_schedule()
            self.optimizer.zero_grad(set_to_none=True)

            # Phase timing only on steps that get logged: the synchronise needed
            # to attribute time would otherwise serialise every step.
            timed = self.rank == 0 and (self.step + 1) % config.log_every == 0
            spent = {"data": 0.0, "forward": 0.0, "backward": 0.0}

            def mark(name: str, since: float) -> float:
                if not timed:
                    return 0.0
                torch.cuda.synchronize()
                now = time.time()
                spent[name] += now - since
                return now

            for micro in range(config.grad_accum):
                phase_started = time.time() if timed else 0.0
                batch = next(iterator)
                phase_started = mark("data", phase_started) or phase_started
                sync = micro == config.grad_accum - 1
                context = (
                    self.model.no_sync()
                    if (not sync and hasattr(self.model, "no_sync"))
                    else _nullcontext()
                )
                with context:
                    loss, metrics = self.compute_loss(batch)
                    phase_started = mark("forward", phase_started) or phase_started
                    (loss / config.grad_accum).backward()
                    mark("backward", phase_started)
                for key, value in metrics.items():
                    accumulated[key] += value
                    counts[key] += 1

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                config.grad_clip,
            )
            self.optimizer.step()
            self.step += 1

            if self.rank == 0 and self.step % config.log_every == 0:
                elapsed = time.time() - started
                averaged = {k: accumulated[k] / max(counts[k], 1) for k in accumulated}
                summary = " ".join(f"{k}={v:.4f}" for k, v in sorted(averaged.items()))
                phases = ""
                if any(spent.values()):
                    phases = " " + " ".join(f"{k}={v:.2f}" for k, v in spent.items())
                log_fn(
                    f"step {self.step}/{config.max_steps} lr={learning_rate:.2e} "
                    f"grad={float(grad_norm):.2f} {summary} "
                    f"({elapsed / config.log_every:.2f}s/step{phases})"
                )
                if self.logger is not None:
                    self.logger.log(
                        {
                            **{f"train/{k}": v for k, v in averaged.items()},
                            "train/lr": learning_rate,
                            "train/grad_norm": float(grad_norm),
                            "train/box_jitter": self._current_jitter(),
                            "train/seconds_per_step": elapsed / config.log_every,
                        },
                        step=self.step,
                    )
                accumulated.clear()
                counts.clear()
                started = time.time()

            visualization = self.config.visualization
            if (
                self.rank == 0
                and visualization.every
                and self.step % visualization.every == 0
            ):
                self.visualize(log_fn=log_fn)
                started = time.time()

            # Evaluation is the one periodic task every rank joins: it is far
            # too slow to leave the rest of the job waiting on rank 0.
            if config.eval_every and self.step % config.eval_every == 0 and self.val_samples:
                metrics = evaluate(
                    self.core,
                    self.val_samples,
                    max_samples=config.eval_samples,
                    log_fn=log_fn if self.rank == 0 else lambda _message: None,
                    rank=self.rank,
                    world_size=self.world_size,
                )
                if self.logger is not None:
                    self.logger.log(
                        {f"val/{k}": v for k, v in metrics.items()}, step=self.step
                    )
                self._keep_if_best(metrics, log_fn=log_fn)
                started = time.time()

            if self.rank == 0 and config.save_every and self.step % config.save_every == 0:
                self.save(os.path.join(config.output_dir, f"step_{self.step}.pt"))

        if self.rank == 0:
            self.save(os.path.join(config.output_dir, "final.pt"))

    # ------------------------------------------------------------------ #
    # visualisation
    # ------------------------------------------------------------------ #

    def visualize(self, log_fn=print) -> None:
        """Render overlays for the current model state and ship them to wandb."""
        settings = self.config.visualization
        images: dict[str, list] = {}
        started = time.time()

        if settings.source in ("train", "both") and self._last_batch is not None:
            try:
                images["viz/teacher_forced"] = visualize_teacher_forced(
                    self.core,
                    self._last_batch,
                    self._last_output,
                    max_samples=settings.num_samples,
                    max_side=settings.max_side,
                    draw_initial=settings.draw_initial,
                )
            except Exception as error:  # visualisation must never kill a run
                log_fn(f"[viz] teacher-forced rendering failed: {error}")

        if settings.source in ("val", "both") and self.val_samples:
            try:
                images["viz/generated"] = visualize_generated(
                    self.core,
                    spread_samples(self.val_samples, settings.num_samples),
                    num_points=self.config.contour.num_points,
                    max_samples=settings.num_samples,
                    max_side=settings.max_side,
                    draw_initial=settings.draw_initial,
                )
            except Exception as error:
                log_fn(f"[viz] generated rendering failed: {error}")

        total = sum(len(panels) for panels in images.values())
        if not total:
            return

        if settings.save_to_disk:
            directory = os.path.join(self.config.train.output_dir, "visualizations")
            for key, panels in images.items():
                save_panels(panels, directory, self.step, key.split("/")[-1])

        if self.logger is not None:
            self.logger.log_images(images, step=self.step)
        log_fn(f"[viz] step {self.step}: {total} panels in {time.time() - started:.1f}s")

    # ------------------------------------------------------------------ #
    # checkpoints
    # ------------------------------------------------------------------ #

    def _keep_if_best(self, metrics: dict[str, float], log_fn=print) -> None:
        """Keep the checkpoint with the best validation metric.

        The training loss is not the selection signal: it can fall while
        generation on unseen images gets worse.
        """
        metric = self.config.train.best_metric
        if not metric or metric not in metrics:
            return
        value = float(metrics[metric])
        if self.best_value is not None and value <= self.best_value:
            return
        self.best_value = value
        self.best_step = self.step
        if self.rank == 0:
            self.save(os.path.join(self.config.train.output_dir, "best.pt"))
            log_fn(f"[best] {metric}={value:.4f} at step {self.step} -> best.pt")
        if self.logger is not None:
            self.logger.log(
                {f"val/best_{metric}": value, "val/best_step": self.step}, step=self.step
            )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Only the trainable slice is stored: the frozen V-L interaction weights
        # stay in their original checkpoint directory.
        trainable = {
            name for name, parameter in self.core.named_parameters() if parameter.requires_grad
        }
        state = {
            name: tensor.detach().cpu()
            for name, tensor in self.core.state_dict().items()
            if name in trainable
        }
        torch.save(
            {
                "model": state,
                "config": self.config.to_dict(),
                "step": self.step,
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def initialize_from(
        self, path: str, keep: str = "", drop: str = "", log_fn=print
    ) -> None:
        """Seed a new run from an old checkpoint's weights and nothing else.

        Distinct from ``load``: the step counter and the optimizer stay at zero,
        because this is the start of a run with its own horizon, not the middle
        of the one that produced the file. Resuming a 12000-step checkpoint into
        a 6000-step schedule would not even enter the training loop.

        ``keep`` filters by prefix, ``drop`` removes by substring afterwards.
        Taking everything is rarely what you want here - see
        ``TrainConfig.init_keep``.

        ``drop`` exists because a submodule worth discarding does not always sit
        under a prefix of its own: the routed cross-attention lives inside
        ``action_decoder.blocks.*``, so no allowlist can keep the decoder while
        dropping it.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["model"]
        prefixes = tuple(p.strip() for p in keep.split(",") if p.strip())
        if prefixes:
            state = {k: v for k, v in state.items() if k.startswith(prefixes)}
            if not state:
                raise ValueError(
                    f"init_keep={keep!r} matched none of the {len(payload['model'])} "
                    f"tensors in {path}"
                )
        patterns = tuple(p.strip() for p in drop.split(",") if p.strip())
        if patterns:
            before = len(state)
            state = {
                k: v for k, v in state.items()
                if not any(pattern in k for pattern in patterns)
            }
            if len(state) == before:
                raise ValueError(
                    f"init_drop={drop!r} matched none of the {before} tensors "
                    f"init_keep selected; a silent no-op here would leave the "
                    f"module it was meant to reset carrying old weights"
                )
        # ``strict=False`` forgives a missing or unexpected key but still raises
        # on a shape mismatch, which is the one thing this path exists to
        # survive: a module that changed width is exactly the "older
        # architecture" case, and refusing the whole file over one tensor would
        # throw away the backbone too.
        current = self.core.state_dict()
        resized = [
            key for key, value in state.items()
            if key in current and current[key].shape != value.shape
        ]
        for key in resized:
            del state[key]

        _, unexpected = self.core.load_state_dict(state, strict=False)
        if resized:
            log_fn(
                f"  {len(resized)} tensor(s) changed shape since the checkpoint and "
                f"start from scratch: {', '.join(sorted(resized)[:4])}"
                + (" ..." if len(resized) > 4 else "")
            )
        log_fn(
            f"initialised {len(state) - len(unexpected)}/{len(payload['model'])} tensors "
            f"from {path} (step {payload.get('step')}); training starts at step 0"
        )
        if unexpected:
            # Not fatal, unlike in ``load``: borrowing weights from an older
            # architecture is the entire point of this path, so a parameter that
            # no longer exists is news rather than an error. It is named because
            # the failure mode is inheriting less than you meant to.
            log_fn(
                f"[init] {len(unexpected)} tensor(s) in the checkpoint no longer "
                f"exist and were skipped: {sorted(unexpected)[:6]}"
            )

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _, unexpected = self.core.load_state_dict(payload["model"], strict=False)
        self.step = int(payload.get("step", 0))
        if unexpected:
            raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
        if "optimizer" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except ValueError as error:
                print(f"[resume] optimizer state not restored: {error}")


def _infinite(loader: Iterable):
    epoch = 0
    while True:
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #


class WandbLogger:
    """Thin wrapper so the trainer never has to know whether wandb is present."""

    def __init__(self, config: ContourVLAConfig) -> None:
        settings = config.wandb
        self.enabled = False
        self.run = None
        if not settings.enabled or settings.mode == "disabled":
            return

        if settings.activate_proxy and settings.mode == "online":
            # Export the proxy variables directly.
            if not os.environ.get("http_proxy") and settings.http_proxy:
                os.environ["http_proxy"] = settings.http_proxy
                os.environ["https_proxy"] = settings.http_proxy
                os.environ.setdefault("no_proxy", settings.no_proxy)
        if settings.api_key:
            os.environ["WANDB_API_KEY"] = settings.api_key
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_START_METHOD", "thread")

        try:
            import wandb
        except ImportError:
            print("[wandb] not installed; logging to stdout only", flush=True)
            return

        try:
            self.run = wandb.init(
                project=settings.project,
                entity=settings.entity or None,
                name=settings.run_name or None,
                group=settings.group or None,
                tags=list(settings.tags) or None,
                mode=settings.mode,
                dir=config.train.output_dir,
                config=config.to_dict(),
                resume="allow",
            )
            self.wandb = wandb
            self.enabled = True
            print(f"[wandb] {self.run.url if settings.mode == 'online' else settings.mode}", flush=True)
        except Exception as error:
            print(f"[wandb] init failed ({error}); continuing without it", flush=True)

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.enabled:
            self.run.log(metrics, step=step)

    def log_images(self, panels: dict[str, list], step: int) -> None:
        if not self.enabled:
            return
        payload = {}
        for key, entries in panels.items():
            if entries:
                payload[key] = [
                    self.wandb.Image(panel.image, caption=panel.caption)
                    for panel in pad_to_common_size(entries)
                ]
        if payload:
            self.run.log(payload, step=step)

    def finish(self) -> None:
        if self.enabled:
            self.run.finish()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def spread_samples(samples, count: int) -> list:
    """``count`` samples drawn evenly across the split, not from its head.

    Samples are ordered by reference and every image carries several references
    with several sentences each, so a prefix is far narrower than it looks: the
    first 200 gRefCOCO validation samples come from just 21 images. A fixed
    stride reaches 200 distinct ones for the same cost, and being fixed rather
    than random keeps the metric comparable from step to step.
    """
    if count <= 0 or not samples:
        return []
    stride = max(len(samples) // count, 1)
    return list(samples[::stride][:count])


@torch.no_grad()
def evaluate(
    model: ContourVLA,
    samples,
    max_samples: int = 200,
    iou_threshold: float = 0.5,
    log_fn=print,
    rank: int = 0,
    world_size: int = 1,
    progress: bool = False,
) -> dict[str, float]:
    """Detection and mask quality on generated (not teacher-forced) outputs.

    Every rank evaluates a slice of the same fixed sample set and the totals are
    summed at the end. Running this on rank 0 alone would leave the others
    parked in a collective for the duration, and box decoding is autoregressive:
    200 samples take a quarter of an hour, comfortably past the point where the
    process group gives up and takes the job with it.

    ``progress`` is off by default because this is called from the training loop
    on every rank, where a bar from each of sixteen of them would interleave into
    noise around the metrics that matter. ``scripts/evaluate.py`` turns it on.
    """
    from PIL import Image

    from .progress import Progress

    was_training = model.training
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    counted = 0

    # Inert while decoding is greedy, which it is by default, but the moment
    # anyone raises the temperature an unseeded metric starts moving between
    # evaluations for reasons that have nothing to do with the weights.
    torch.manual_seed(model.config.inference.seed + rank)

    chosen = spread_samples(samples, max_samples)[rank::world_size]
    bar = Progress(len(chosen), interval=50) if progress else None
    for sample in chosen:
        if not os.path.exists(sample.image_path):
            if bar is not None:
                bar.update()
            continue
        image = Image.open(sample.image_path).convert("RGB")
        result = model.predict(image, sample.expression)

        predicted = np.asarray(result["boxes"], dtype=np.float32).reshape(-1, 4)
        truth = np.asarray(sample.boxes, dtype=np.float32).reshape(-1, 4)

        if bar is not None:
            done = max(counted, 1)
            bar.update(f"mask_iou {totals['mask_iou'] / done:.4f}")

        if truth.shape[0] == 0:
            totals["no_target_accuracy"] += float(predicted.shape[0] == 0)
            totals["no_target_count"] += 1
            # GRES convention: a no-target sample scores 1 exactly when nothing
            # is predicted, so gIoU can rank checkpoints on refusal and masks at once.
            totals["giou"] += float(predicted.shape[0] == 0)
            continue

        iou = box_iou(predicted, truth)
        matched = iou.max(axis=0) if predicted.shape[0] else np.zeros(truth.shape[0])
        totals["box_recall"] += float((matched >= iou_threshold).mean())
        totals["box_iou"] += float(matched.mean())

        # Recall on its own is blind to over-prediction, and rewards it.
        hits = match_boxes_one_to_one(predicted, truth, iou_threshold)
        totals["true_positives"] += hits
        totals["predicted_boxes"] += predicted.shape[0]
        totals["target_boxes"] += truth.shape[0]

        contours = result.get("contours")
        if contours is not None and predicted.shape[0]:
            height = int(min(sample.height, 640))
            width = max(int(round(sample.width * height / max(sample.height, 1))), 1)
            # Each contour carries only as many real points as its object was
            # given; the rest of the row is padding and would close the polygon
            # through the origin.
            counts = result.get("point_counts")
            settings = model.config.contour
            union_pred = np.zeros((height, width), dtype=bool)
            for index, contour in enumerate(contours):
                length = int(counts[index]) if counts is not None else len(contour)
                ring = contour[:length]
                if settings.smooth_harmonics or settings.render_points:
                    ring = smooth_contour(
                        ring, settings.smooth_harmonics, settings.render_points
                    )
                union_pred |= contour_to_mask(ring, height, width)
            union_true = np.zeros((height, width), dtype=bool)
            for segmentation in sample.segmentations:
                ring = segmentation_to_contour(segmentation, 256, sample.width, sample.height)
                if ring is not None:
                    union_true |= contour_to_mask(ring, height, width)
            intersection = float((union_pred & union_true).sum())
            union = float((union_pred | union_true).sum())
            totals["mask_iou"] += intersection / max(union, 1.0)
            totals["giou"] += intersection / max(union, 1.0)

        counted += 1

    if bar is not None:
        done = max(counted, 1)
        bar.close(f"mask_iou {totals['mask_iou'] / done:.4f}")

    keys = (
        "box_recall", "box_iou", "mask_iou", "no_target_accuracy", "no_target_count",
        "true_positives", "predicted_boxes", "target_boxes", "giou",
    )
    tally = torch.tensor(
        [totals[key] for key in keys] + [float(counted)],
        dtype=torch.float64,
        device=next(model.parameters()).device,
    )
    if world_size > 1:
        torch.distributed.all_reduce(tally)
    summed = dict(zip(keys, tally[:-1].tolist()))
    counted = tally[-1].item()

    metrics: dict[str, float] = {}
    if counted:
        for key in ("box_recall", "box_iou", "mask_iou"):
            metrics[key] = summed[key] / counted
        metrics["boxes_per_image"] = summed["predicted_boxes"] / counted
        metrics["target_boxes_per_image"] = summed["target_boxes"] / counted
    # Pooled rather than averaged per image, so an image with eight objects
    # weighs eight times an image with one - the same convention as detection
    # benchmarks, and it stops a single crowded failure hiding in the mean.
    if summed["predicted_boxes"]:
        metrics["box_precision"] = summed["true_positives"] / summed["predicted_boxes"]
    if summed["target_boxes"]:
        pooled_recall = summed["true_positives"] / summed["target_boxes"]
        precision = metrics.get("box_precision", 0.0)
        metrics["box_f1"] = (
            2 * precision * pooled_recall / max(precision + pooled_recall, 1e-9)
        )
    if summed["no_target_count"]:
        metrics["no_target_accuracy"] = (
            summed["no_target_accuracy"] / summed["no_target_count"]
        )
    metrics["eval_samples"] = counted + summed["no_target_count"]
    if metrics["eval_samples"]:
        metrics["giou"] = summed["giou"] / metrics["eval_samples"]
    log_fn(json.dumps(metrics, indent=2))
    if was_training:
        model.train()
    return metrics
