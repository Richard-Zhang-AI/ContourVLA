"""Turning per-object credit into a per-token advantage.

GRPO gives every token of a rollout the same advantage. That is not a detail of
the implementation but a proved property of the objective: under an output-only
reward the gradient matrix has effective rank 2 whatever the group size
(arXiv 2606.29238). A rollout that emits one right box and one wrong box then
trains both identically.

The transport plan in ``transport.py`` says which box earned what. This module
spends that on the tokens that produced each box, leaving the rest of the
sequence on the ordinary group-relative signal.
"""

from __future__ import annotations

import torch


def group_advantage(rewards: torch.Tensor, epsilon: float = 0.02) -> torch.Tensor:
    """Standardise a group's rewards: ``[G]``.

    A group whose rollouts all scored alike carries no information, and dividing
    its near-zero spread would manufacture some. Those groups return zero, which
    is the honest answer and also what stops a batch of uniformly-failed hard
    cases from producing enormous updates.

    ``epsilon`` has to be set against the reward's own resolution, not merely
    above zero. A threshold near zero standardises rasterisation jitter into a
    unit-scale advantage.
    """
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    spread = rewards.std(unbiased=False)
    if float(spread) < epsilon:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / spread


def per_token_advantage(
    sequence_advantage: float,
    credits: torch.Tensor,
    spans: list[tuple[int, int]],
    length: int,
    blend: float = 0.5,
) -> torch.Tensor:
    """Advantage for each answer token: ``[length]``.

    Every token starts on the sequence-level advantage, and tokens inside a box
    are then tilted by how that box did relative to its siblings. A rollout with
    one good and one bad box pushes them in opposite directions instead of the
    same one.

    ``blend`` is how much of the signal is per-object. At 0 this is standard
    GRPO; at 1 the sequence-level term is gone and a rollout that scored badly
    overall could still reinforce its least-bad box, which is not wanted. The
    within-rollout term is centred, so it redistributes the advantage rather
    than adding to it - the total pull on a rollout is unchanged and only its
    allocation moves.
    """
    advantage = torch.full((length,), float(sequence_advantage), dtype=torch.float32)
    if not spans or credits.numel() == 0 or blend <= 0:
        return advantage

    clipped = [(max(0, s), min(length, e)) for s, e in spans[: credits.numel()]]
    widths = torch.tensor(
        [float(max(e - s, 0)) for s, e in clipped], dtype=torch.float32
    )
    if float(widths.sum()) <= 0:
        return advantage

    # Centred by token count, not by box, so the tilt sums to zero over the
    # tokens it is actually applied to. Boxes differ in length - a coordinate
    # that rounds to a shorter token is enough - and centring per box would
    # leave a residue that shifts the rollout's total pull. The first draft
    # scaled the base term by (1 - blend) as well, which halved the pull outright
    # and would have shown up only as a mysteriously slow learning rate.
    weighted_mean = float((widths * credits).sum() / widths.sum())
    centred = credits - weighted_mean
    scale = centred.abs().max()
    if float(scale) < 1e-6:
        return advantage  # every box did equally well; nothing to redistribute

    tilt = centred / scale
    for (start, end), amount in zip(clipped, tilt.tolist()):
        if end > start:
            advantage[start:end] = sequence_advantage + blend * abs(
                sequence_advantage
            ) * amount
    return advantage


def scale_negative(advantage: torch.Tensor, is_negative: bool, alpha: float) -> torch.Tensor:
    """Damp the pull of no-target samples.

    Refusing is an easier task than localising, so an undamped no-target term
    pulls the policy toward refusing everything.
    """
    return advantage * alpha if is_negative else advantage


def policy_gradient_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantage: torch.Tensor,
    reference_log_probs: torch.Tensor,
    clip: float = 0.2,
    kl_weight: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate plus a KL anchor, averaged over tokens.

    The anchor keeps the policy from drifting off the reference it started from.

    KL is the k3 estimator (exp(d) - d - 1, d = ref - policy): unbiased,
    non-negative, and far lower variance than the naive difference, which is
    what keeps a single surprising token from swamping the batch.
    """
    if log_probs.numel() == 0:
        zero = log_probs.sum() * 0.0
        return zero, {"kl": 0.0, "clipped": 0.0}

    ratio = (log_probs - old_log_probs).exp()
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    surrogate = -torch.min(ratio * advantage, clipped * advantage).mean()

    delta = reference_log_probs - log_probs
    kl = (delta.exp() - delta - 1).mean()

    total = surrogate + kl_weight * kl
    return total, {
        "kl": float(kl.detach()),
        "clipped": float(((ratio < 1 - clip) | (ratio > 1 + clip)).float().mean()),
    }


def instance_advantage(deltas, active, floor: float = 0.001):
    """Standardise per-instance action improvements across one group: ``[[M_r]]``.

    ``deltas[r][i]`` is how much better rollout ``r``'s sampled contour for box
    ``i`` scored than the deterministic contour for the same box. That
    baseline depends on the answer but not on the sampled actions, so the
    estimate stays unbiased while the variance from *which object was chosen*
    - the grounding decision, which dominates the sequence reward - drops out
    of the action gradient entirely.

    Only ``active`` instances (ones that touch a target at all) are scored: a
    box on nothing earns zero whatever its contour does, and giving it a
    centred advantage would only add noise. Below ``floor`` the spread is
    rasterisation jitter rather than a signal, and the whole group gets zero.
    """
    pooled = [d[a] for d, a in zip(deltas, active) if d is not None and d.numel()]
    values = torch.cat(pooled) if pooled else torch.zeros(0)
    zeros = [None if d is None else torch.zeros_like(d) for d in deltas]
    if values.numel() < 2:
        return zeros, 0.0
    spread = float(values.std(unbiased=False))
    if spread < floor:
        return zeros, spread
    mean = values.mean()
    return [
        None if d is None else ((d - mean) / spread) * a.to(d.dtype)
        for d, a in zip(deltas, active)
    ], spread


def action_policy_loss(log_probs, old_log_probs, point_mask, instance_advantages, clip=0.2):
    """Clipped surrogate over valid 2-D actions, not a trajectory product ratio.

    Densities are [M,K,T,P]. Every valid action of instance ``i`` carries that
    instance's advantage; the mean is over valid actions.
    """
    if log_probs.shape != old_log_probs.shape or log_probs.ndim != 4:
        raise ValueError("action densities must have matching [M,K,T,P] shapes")
    if point_mask.shape != (log_probs.shape[0], log_probs.shape[-1]):
        raise ValueError("point mask must match the sampled action instances and points")
    if instance_advantages.numel() != log_probs.shape[0]:
        raise ValueError("one advantage is required per action instance")
    valid = point_mask[:, None, None, :].expand_as(log_probs).bool()
    current = log_probs.float()[valid]
    old = old_log_probs.detach().to(log_probs).float()[valid]
    if not current.numel():
        return current.sum() * 0.0, {"action_clipped": 0.0, "action_logprob_delta": 0.0}
    per_action = instance_advantages.detach().float().to(log_probs.device)
    advantage = per_action[:, None, None, None].expand_as(log_probs)[valid]
    ratio = (current - old).exp()
    loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
    return loss, {
        "action_clipped": float(((ratio < 1 - clip) | (ratio > 1 + clip)).float().mean()),
        "action_logprob_delta": float((current - old).detach().abs().mean()),
    }
