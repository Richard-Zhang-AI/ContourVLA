"""Joint score-function training of grounding answers and continuous actions.

The joint policy factorizes into the answer likelihood and action densities at
saved contour states, and each factor gets its own advantage:

* grounding is priced on the *deterministic* contour of each rollout - the mask
  inference would draw - so the answer tokens see exactly the reward the
  grounding-only stage used. Pricing them on sampled contours instead let
  action noise decide which of several identical answers was reinforced
  (through the per-box credit tilt) and turned noise-sized reward spreads into
  unit-scale grounding updates;
* each sampled action is scored against the deterministic contour for the same
  box: ``delta_i = credit(sampled_i) - credit(mean_i)`` under the deterministic
  transport plan. The baseline depends on the answer but not on the sampled
  actions, so the estimate is unbiased, and the variance from which object was
  chosen - which dominates the sequence reward - no longer reaches the actions.

Rasterized rewards are always detached. Grounding keeps the untempered replay
likelihood; the KL reference is whatever ``as_reference`` provides (see
``rl.reference``). No temperature/top-p importance correction is claimed.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..contour.geometry import contour_to_mask, smooth_contour
from ..tokens import build_prompt
from .advantage import (
    action_policy_loss,
    group_advantage,
    instance_advantage,
    per_token_advantage,
    policy_gradient_loss,
    scale_negative,
)
from .loop import hidden_states, mask_quality, target_masks
from .rollout import answer_labels, as_reference, refusal_rollout, sample_group
from .transport import credit_under_plan, object_credit_and_plan, sequence_reward


def prediction_masks(prediction, counts, height, width, settings):
    if prediction is None:
        return []
    contours = prediction.final.detach().float().clamp(0, 1).cpu().numpy()
    masks = []
    for ring, count in zip(contours, counts):
        ring = np.asarray(ring[:int(count)], dtype=np.float32)
        if settings.smooth_harmonics or settings.render_points:
            ring = smooth_contour(ring, settings.smooth_harmonics, settings.render_points)
        masks.append(contour_to_mask(ring, height, width))
    return masks


@torch.no_grad()
def collect_joint_rollouts(model, encoded, prompt, expression_chars, sample, rollouts, settings):
    """Price every rollout and freeze its replay record before any update."""
    height = int(min(sample.height, 640))
    width = max(int(round(sample.width * height / max(sample.height, 1))), 1)
    contour_settings = model.config.contour
    targets = target_masks(sample, height, width, contour_settings)
    for rollout in rollouts:
        result = model.contours_for_answer(
            encoded[0], encoded[1], prompt, rollout.answer, expression_chars,
            return_tensors=True, action_std=settings.action_std, deterministic_too=True,
        )
        _, labels = answer_labels(
            model.backbone.tokenizer, prompt, rollout.answer, model.backbone.device
        )
        rollout.old_log_probs = model.backbone.answer_log_probs(
            result["backbone"].hidden_states, labels
        ).detach().cpu()
        rollout.spans = result["box_spans"]
        prediction = result["prediction"]
        sampled = prediction_masks(prediction, result["point_counts"], height, width,
                                   contour_settings)
        deterministic = prediction_masks(result.get("mean_prediction"), result["point_counts"],
                                         height, width, contour_settings)
        if prediction is not None:
            rollout.action_latents = prediction.action_latents.detach().cpu()
            rollout.action_log_probs = prediction.action_log_probs.detach().cpu()
            rollout.action_point_mask = result["point_mask"].detach().cpu()
            if targets:
                rollout.replay_context = result.get("context")
        if len(sampled) != len(rollout.spans):
            raise RuntimeError("reward instances and grounding spans are not aligned")
        if targets:
            quality = mask_quality(deterministic, targets)
            earned, found, plan = object_credit_and_plan(
                quality, settings.epsilon, settings.dustbin
            )
            rollout.reward = float(sequence_reward(earned, found))
            rollout.credits = earned.detach().cpu()
            sampled_quality = mask_quality(sampled, targets)
            sampled_earned, sampled_found = object_credit_and_plan(
                sampled_quality, settings.epsilon, settings.dustbin
            )[:2]
            rollout.sampled_reward = float(sequence_reward(sampled_earned, sampled_found))
            if plan is not None:
                through_plan = credit_under_plan(plan, sampled_quality)
                rollout.action_deltas = (through_plan - earned).detach().cpu()
                rollout.action_active = ((through_plan > 1e-6) | (earned > 1e-6)).cpu()
        else:
            rollout.reward = 1.0 / (1.0 + len(sampled))
            rollout.sampled_reward = rollout.reward
            rollout.credits = torch.zeros(len(sampled))
        del result, prediction
        rollout.metrics = {"boxes": len(sampled), "targets": len(targets)}


def joint_rl_step(model, image, sample, settings):
    """Accumulate one group's grounding and action gradients; caller steps AdamW.

    eval mode disables dropout but not autograd. Both reward collection and
    gradient replay use identical states, masks, routes, and normalization.
    """
    device = model.backbone.device
    tokenizer = model.backbone.tokenizer
    negative = len(sample.boxes) == 0
    actions_reach_backbone = bool(model.config.train.segmentation_updates_backbone)
    was_training = model.training
    model.eval()
    try:
        started = time.perf_counter()
        encoded = model.encode_image(image)
        prompt, expression_chars = build_prompt(
            sample.expression, encoded[2], model.config.data.prompt_template, with_span=True
        )
        rollouts = sample_group(model, image, sample.expression, settings.group,
                                settings.temperature, settings.top_p, settings.temperature_jitter)
        if negative and settings.force_refusal:
            rollouts[-1] = refusal_rollout(model)
        sampled_at = time.perf_counter()
        collect_joint_rollouts(model, encoded, prompt, expression_chars, sample, rollouts, settings)
        collected_at = time.perf_counter()

        rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)
        advantages = group_advantage(rewards, settings.advantage_floor)
        # No-target reward depends on the answer's count only, not on any contour.
        if negative:
            action_advantages, action_spread = [None] * len(rollouts), 0.0
        else:
            action_advantages, action_spread = instance_advantage(
                [r.action_deltas for r in rollouts],
                [r.action_active for r in rollouts],
                settings.action_advantage_floor,
            )

        stats = dict(kl=0.0, clipped=0.0, grounding_loss=0.0, action_loss=0.0,
                     action_clipped=0.0, action_logprob_delta=0.0, action_instances=0.0)
        grounded = acted = 0
        total = 0.0
        for rollout, advantage, instance_adv in zip(rollouts, advantages.tolist(),
                                                    action_advantages):
            grounding = abs(advantage) >= 1e-8 and rollout.old_log_probs.numel() > 0
            acting = (
                instance_adv is not None and rollout.action_latents is not None
                and bool((instance_adv.abs() > 0).any())
            )
            if not (grounding or acting):
                continue
            input_ids, labels = answer_labels(tokenizer, prompt, rollout.answer, device)

            # Before any graph is built: an "initial" reference swaps the
            # trainable tensors in place, which a pending backward would reject.
            reference = None
            if grounding:
                if settings.kl_weight:
                    with torch.no_grad(), as_reference(model):
                        reference = model.backbone.answer_log_probs(
                            hidden_states(model, encoded, input_ids), labels
                        )
                else:
                    reference = rollout.old_log_probs.to(device)

            prediction = point_mask = hidden = None
            if acting and not grounding and rollout.replay_context is not None:
                # Nothing needs the backbone's graph: rebuild only the contour
                # branch from the read-outs saved at collection.
                context = rollout.replay_context
                prediction = model.evolve_from_context(
                    context, settings.action_std, rollout.action_latents.to(device)
                )
                point_mask = context["point_mask"]
            elif acting:
                result = model.contours_for_answer(
                    encoded[0], encoded[1], prompt, rollout.answer, expression_chars,
                    return_tensors=True, action_std=settings.action_std,
                    action_latents=rollout.action_latents,
                    backbone_grad=grounding or actions_reach_backbone,
                )
                if result["box_spans"] != rollout.spans:
                    raise RuntimeError("action replay does not match its collection record")
                prediction, point_mask = result["prediction"], result["point_mask"]
                hidden = result["backbone"].hidden_states
                del result
            else:
                hidden = hidden_states(model, encoded, input_ids)

            loss = None
            if grounding:
                log_probs = model.backbone.answer_log_probs(hidden, labels)
                if log_probs.shape != rollout.old_log_probs.shape:
                    raise RuntimeError("grounding replay does not match its collection record")
                shaped = per_token_advantage(advantage, rollout.credits, rollout.spans,
                                             log_probs.numel(), settings.credit_blend).to(device)
                shaped = scale_negative(shaped, negative, settings.negative_alpha)
                grounding_loss, extra = policy_gradient_loss(
                    log_probs, rollout.old_log_probs.to(device), shaped,
                    reference, settings.clip, settings.kl_weight,
                )
                loss = grounding_loss
                stats["grounding_loss"] += float(grounding_loss.detach())
                for key, value in extra.items():
                    stats[key] += value
                grounded += 1

            if acting:
                if not torch.equal(point_mask.cpu(), rollout.action_point_mask):
                    raise RuntimeError("action point budget changed during replay")
                action_loss, action_stats = action_policy_loss(
                    prediction.action_log_probs, rollout.action_log_probs.to(device),
                    point_mask, instance_adv, settings.clip,
                )
                weighted = settings.action_loss_weight * action_loss
                loss = weighted if loss is None else loss + weighted
                stats["action_loss"] += float(action_loss.detach())
                stats["action_instances"] += float(prediction.final.shape[0])
                for key, value in action_stats.items():
                    stats[key] += value
                acted += 1

            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite joint RL loss")
            (loss / settings.group).backward()
            total += abs(float(loss.detach()))
            rollout.replay_context = None
            del loss, hidden, prediction

        grounding_keys = ("kl", "clipped", "grounding_loss")
        action_keys = ("action_loss", "action_clipped", "action_logprob_delta", "action_instances")
        for key in grounding_keys:
            stats[key] /= max(grounded, 1)
        for key in action_keys:
            stats[key] /= max(acted, 1)
        sampled_rewards = torch.tensor([r.sampled_reward for r in rollouts], dtype=torch.float32)
        return dict(
            pull=total / max(grounded + acted, 1),
            advantage=float(advantages.abs().mean()),
            reward=float(rewards.mean()),
            reward_best=float(rewards.max()),
            reward_spread=float(rewards.std(unbiased=False)),
            reward_sampled=float(sampled_rewards.mean()),
            action_spread=float(action_spread),
            rollouts_used=float(grounded),
            action_rollouts=float(acted),
            group_used=float(grounded > 0),
            action_group_used=float(acted > 0),
            negative=float(negative),
            time_sample=sampled_at - started,
            time_collect=collected_at - sampled_at,
            time_replay=time.perf_counter() - collected_at,
            **stats,
        )
    finally:
        model.train(was_training)
