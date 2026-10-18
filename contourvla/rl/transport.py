"""Credit assignment for a set of predicted objects, by optimal transport.

A rollout emits a set of boxes; the annotation is a set of objects. Deciding
which prediction answers for which target is an assignment problem, and the hard
version of it - Hungarian matching, as DETR and VisionReasoner use - is the
zero-temperature case of optimal transport.

Hard matching is the wrong end of that family for this task. Many failures have
the right number of boxes and the wrong object, so the model is choosing among
several plausible candidates. A permutation gives each prediction full credit or
none, and says nothing about how nearly right a choice was.

Entropic regularisation replaces the permutation with a transport plan, and the
plan's rows are fractional responsibility. The regulariser has an exact reading:
the plan is the Gibbs distribution over couplings at temperature ``epsilon``,
minimising cost minus entropy. At ``epsilon -> 0`` it recovers hard matching; at
``epsilon -> inf`` it spreads credit uniformly and says nothing.

Sets differ in size, and both surplus predictions and unfound targets have to
cost something. That is handled with a dustbin row and column in the manner of
SuperGlue, so unmatched mass has somewhere to go at a stated price rather than
being forced onto a real object.
"""

from __future__ import annotations

import torch


def sinkhorn_plan(
    quality: torch.Tensor,
    epsilon: float,
    dustbin: float,
    iterations: int = 100,
) -> torch.Tensor:
    """Transport plan between predictions and targets: ``[M + 1, N + 1]``.

    Args:
        quality: ``[M, N]`` in [0, 1]; how well prediction ``i`` answers for
            target ``j``. Mask IoU here, not box IoU - see ``object_credit``.
        epsilon: entropic temperature. Small is close to hard matching.
        dustbin: the quality assigned to leaving something unmatched. A
            prediction is sent here rather than onto a target it fits worse
            than this.
        iterations: Sinkhorn steps. Convergence is geometric; 100 is far more
            than enough at the temperatures used here.

    Returns:
        A plan whose last row and column absorb unmatched mass. Rows sum to the
        prediction marginal and columns to the target marginal, both by
        construction rather than approximately.
    """
    predictions, targets = quality.shape
    device, dtype = quality.device, torch.float32

    # Work in log space throughout: at the temperatures that make the plan
    # nearly hard, exponentials of cost/epsilon overflow long before the plan does
    # anything interesting.
    # The whole augmented band, corner included, sits at the dustbin value. The
    # corner is where the slack in the marginals is supposed to end up - the
    # dustbin row must shed N units and the dustbin column must absorb M - and
    # making it any less attractive than the rest of the band pushes that slack
    # onto real predictions instead, quietly diluting every credit.
    score = torch.full(
        (predictions + 1, targets + 1), float(dustbin), device=device, dtype=dtype
    )
    score[:predictions, :targets] = quality.to(dtype)

    log_plan = score / epsilon

    # SuperGlue's marginals: every real row and column carries unit mass, and
    # each dustbin carries as much as the opposite side could possibly dump.
    total = predictions + targets
    log_mu = torch.cat([
        torch.zeros(predictions, device=device, dtype=dtype),
        torch.tensor([float(targets)], device=device, dtype=dtype).log(),
    ]) - torch.tensor(float(total), device=device, dtype=dtype).log()
    log_nu = torch.cat([
        torch.zeros(targets, device=device, dtype=dtype),
        torch.tensor([float(predictions)], device=device, dtype=dtype).log(),
    ]) - torch.tensor(float(total), device=device, dtype=dtype).log()

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(log_plan + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(log_plan + u[:, None], dim=0)
    return (log_plan + u[:, None] + v[None, :]).exp() * total


def object_credit(
    quality: torch.Tensor,
    epsilon: float = 0.1,
    dustbin: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(per prediction, per target)`` credit in [0, 1].

    The per-prediction figure is what makes this worth doing. GRPO gives every
    token of a rollout the same advantage - a result proved in arXiv 2606.29238,
    where the gradient matrix collapses to effective rank 2 whatever the group
    size - so a rollout with one right box and one wrong box trains both
    identically. These numbers say which box earned what, and the box tokens can
    be credited separately.

    The per-target figure is the recall side: a target that nothing was
    transported to scores zero, which is what a missed object should cost.
    """
    earned, found, _ = object_credit_and_plan(quality, epsilon, dustbin)
    return earned, found


def object_credit_and_plan(
    quality: torch.Tensor,
    epsilon: float = 0.1,
    dustbin: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """``object_credit`` plus the plan it was read from (``None`` when a side is empty)."""
    predictions, targets = quality.shape
    if predictions == 0 or targets == 0:
        return quality.new_zeros(predictions), quality.new_zeros(targets), None

    plan = sinkhorn_plan(quality, epsilon, dustbin)
    real = plan[:predictions, :targets]

    # Divided by the row's whole mass including the dustbin, so a prediction
    # that was mostly discarded is scored as mostly worthless rather than being
    # judged only on the sliver that found a home.
    earned = credit_under_plan(plan, quality)
    found = (real * quality).sum(dim=0) / plan[:, :targets].sum(dim=0).clamp(min=1e-9)
    return earned, found, plan


def credit_under_plan(plan: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """Per-prediction credit of ``quality`` read through a fixed ``plan``: ``[M]``.

    Holding the assignment fixed is what lets two mask sets for the same boxes
    be compared one instance at a time: re-solving the plan for the second set
    would let every other instance's noise move this instance's credit too.
    """
    predictions, targets = quality.shape
    real = plan[:predictions, :targets]
    return (real * quality).sum(dim=1) / plan[:predictions].sum(dim=1).clamp(min=1e-9)


def sequence_reward(
    earned: torch.Tensor, found: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """A soft F1 over the two sides of the transport plan.

    F-shaped and not recall, deliberately. A recall reward is gamed by emitting
    more boxes. Precision alone is gamed by emitting one. Only something that
    punishes both keeps the count honest.
    """
    if earned.numel() == 0 or found.numel() == 0:
        return earned.new_zeros(())
    precision, recall = earned.mean(), found.mean()
    weight = beta * beta
    return (1 + weight) * precision * recall / (weight * precision + recall).clamp(min=1e-9)
