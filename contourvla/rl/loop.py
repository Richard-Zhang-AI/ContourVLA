"""RL dispatcher and legacy grounding-only reward helpers.

With joint_training enabled, rl_step delegates to joint.py, which samples and
scores continuous actions and backpropagates into both policies. The legacy
path below retains deterministic contour scoring.

One image in the legacy path:

  1. sample ``group`` answers, plus a forced refusal when the image has no target
  2. replay each answer through the frozen contour branch to get real masks
  3. score prediction-target pairs by mask IoU and solve the transport plan
  4. sequence reward = soft F1 of the plan's two margins; per-box credit = its row
  5. advantage = group-relative on the sequence, tilted per box on its own tokens
  6. clipped surrogate plus a KL anchor against the LoRA-disabled reference

The contour branch is frozen throughout. This stage moves the detector only, and
holding the segmenter still is what makes the reward a fixed function of the
boxes rather than a target that moves while the policy chases it.
"""

from __future__ import annotations

import numpy as np
import torch

from ..contour.geometry import contour_to_mask, segmentation_to_contour, smooth_contour
from ..data.datasets import build_prompt
from .advantage import (
    group_advantage,
    per_token_advantage,
    policy_gradient_loss,
    scale_negative,
)
from .rollout import Rollout, answer_labels, as_reference, refusal_rollout, sample_group
from .transport import object_credit, sequence_reward


def target_masks(sample, height: int, width: int, settings) -> list[np.ndarray]:
    """One mask per annotated object, at the raster the metric is reported on."""
    masks = []
    for segmentation in sample.segmentations:
        ring = segmentation_to_contour(
            segmentation, 256, sample.width, sample.height,
            mode=settings.gt_source, raster_side=settings.mask_raster_side,
        )
        masks.append(
            contour_to_mask(ring, height, width) if ring is not None
            else np.zeros((height, width), dtype=bool)
        )
    return masks


@torch.no_grad()
def rollout_masks(model, encoded, prompt: str, answer: str, height: int, width: int):
    """Replay one rollout's answer through the contour branch.

    The rollout's own answer is used rather than its parsed boxes, so what gets
    priced is exactly what the policy emitted, down to the token. The reward is
    mask IoU rather than box IoU: a contour grown from a box on the wrong object
    barely overlaps the right one even when the two boxes do.
    """
    pixel_values, image_grid_hws, _ = encoded
    result = model.contours_for_answer(pixel_values, image_grid_hws, prompt, answer)
    if not result:
        return []

    settings = model.config.contour
    counts = result["point_counts"]
    masks = []
    for index, contour in enumerate(result["contours"]):
        ring = np.asarray(contour[: int(counts[index])], dtype=np.float32)
        if settings.smooth_harmonics or settings.render_points:
            ring = smooth_contour(ring, settings.smooth_harmonics, settings.render_points)
        masks.append(contour_to_mask(ring, height, width))
    return masks


def mask_quality(predictions, targets) -> torch.Tensor:
    """``[M, N]`` mask IoU between every prediction and every target."""
    quality = torch.zeros(len(predictions), len(targets), dtype=torch.float32)
    for i, prediction in enumerate(predictions):
        for j, target in enumerate(targets):
            union = float((prediction | target).sum())
            if union > 0:
                quality[i, j] = float((prediction & target).sum()) / union
    return quality


def price_rollouts(model, encoded, prompt, sample, rollouts: list[Rollout], settings) -> None:
    """Fill in ``reward`` and ``credits`` on each rollout, in place."""
    height = int(min(sample.height, 640))
    width = max(int(round(sample.width * height / max(sample.height, 1))), 1)
    targets = target_masks(sample, height, width, model.config.contour)

    for rollout in rollouts:
        predictions = rollout_masks(
            model, encoded, prompt, rollout.answer, height, width
        )
        if not targets:
            # A no-target image: the only right answer is to emit nothing, so the
            # reward is 1 exactly when the rollout does and decays with however
            # many boxes it invented instead.
            rollout.reward = 1.0 / (1.0 + len(predictions))
            rollout.credits = torch.zeros(len(predictions))
        else:
            earned, found = object_credit(
                mask_quality(predictions, targets),
                epsilon=settings.epsilon,
                dustbin=settings.dustbin,
            )
            rollout.reward = float(sequence_reward(earned, found))
            rollout.credits = earned
        rollout.metrics = {"boxes": len(rollout.boxes), "targets": len(targets)}


def hidden_states(model, encoded, input_ids: torch.Tensor) -> torch.Tensor:
    """Backbone hidden states for a replayed sequence, with gradient.

    Box pooling is asked for over an empty set: this path exists to score tokens,
    and the instance features it would otherwise compute belong to the contour
    branch, which is frozen in this stage.
    """
    pixel_values, image_grid_hws, _ = encoded
    device = model.backbone.device
    return model.backbone(
        pixel_values=pixel_values,
        image_grid_hws=image_grid_hws,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        box_token_index=torch.zeros((1, 0, 6), dtype=torch.long, device=device),
        box_token_mask=torch.zeros((1, 0), dtype=torch.bool, device=device),
        labels=None,
    ).hidden_states


def rl_step(model, image, sample, settings) -> dict[str, float]:
    """One image: sample a group, price it, and accumulate the policy gradient.

    Gradients are left on the parameters; the caller owns the optimiser step so
    that several images can be accumulated into one update.
    """
    if settings.joint_training:
        from .joint import joint_rl_step
        return joint_rl_step(model, image, sample, settings)
    device = model.backbone.device
    tokenizer = model.backbone.tokenizer
    negative = len(sample.boxes) == 0

    encoded = model.encode_image(image)
    prompt = build_prompt(
        sample.expression, encoded[2], model.config.data.prompt_template
    )

    was_training = model.training
    model.eval()
    rollouts = sample_group(
        model, image, sample.expression, settings.group,
        settings.temperature, settings.top_p, settings.temperature_jitter,
    )
    if negative and settings.force_refusal:
        # Without this the group is all boxes, every rollout scores alike, the
        # advantage vanishes and refusal is never learned. RC-GRPO's
        # forced-rejection rollout; the reason it is needed here is the same.
        rollouts[-1] = refusal_rollout(model)
    price_rollouts(model, encoded, prompt, sample, rollouts, settings)

    rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)
    advantages = group_advantage(rewards, settings.advantage_floor)

    model.train(was_training)
    total, stats, counted = 0.0, {"kl": 0.0, "clipped": 0.0}, 0
    for rollout, advantage in zip(rollouts, advantages.tolist()):
        if abs(advantage) < 1e-8:
            continue  # a group that scored uniformly says nothing; skip the pass
        input_ids, labels = answer_labels(tokenizer, prompt, rollout.answer, device)

        # Reference first: an "initial" reference swaps the trainable tensors in
        # place, which is only safe while no graph is holding on to them.
        with torch.no_grad(), as_reference(model):
            reference = model.backbone.answer_log_probs(
                hidden_states(model, encoded, input_ids), labels
            )
        log_probs = model.backbone.answer_log_probs(
            hidden_states(model, encoded, input_ids), labels
        )
        if log_probs.numel() == 0:
            continue

        shaped = per_token_advantage(
            advantage, rollout.credits, rollout.spans,
            log_probs.numel(), blend=settings.credit_blend,
        ).to(device)
        shaped = scale_negative(shaped, negative, settings.negative_alpha)

        loss, extra = policy_gradient_loss(
            log_probs, log_probs.detach(), shaped, reference,
            clip=settings.clip, kl_weight=settings.kl_weight,
        )
        (loss / settings.group).backward()
        # Absolute, and accumulated per rollout. Summing the signed surrogate
        # over a group cancels toward zero.
        total += abs(float(loss.detach()))
        for key, value in extra.items():
            stats[key] += value
        counted += 1

    if counted:
        stats = {key: value / counted for key, value in stats.items()}
    return {
        # The surrogate summed over a group is ~0 by construction - advantages
        # are centred and the ratio is exactly 1 on the first pass over fresh
        # rollouts - so reporting it would print zero however training is going.
        # The magnitude is what says whether there is anything to learn from.
        "pull": total / max(counted, 1),
        "advantage": float(advantages.abs().mean()),
        "reward": float(rewards.mean()),
        "reward_best": float(rewards.max()),
        "reward_spread": float(rewards.std(unbiased=False)),
        "rollouts_used": float(counted),
        # 1 when the group disagreed enough to be worth learning from, 0 when it
        # was floored. The share of steps that clear the floor is the honest
        # measure of how much of this stage is doing anything.
        "group_used": float(counted > 0),
        "negative": float(negative),
        **stats,
    }
