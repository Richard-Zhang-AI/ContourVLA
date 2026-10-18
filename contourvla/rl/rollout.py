"""Sampling a group of answers, and finding each box inside the token stream.

Two things the rest of the RL loop needs and cannot get from ``predict``: the
token span each box occupies, so credit can be spent where the box was decided,
and a way to put a refusal into the group for samples that have no target.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from ..models.lora import LoRALinear
from ..tokens import NONE_TOKEN, find_box_token_spans

BOX = re.compile(r"<box>.*?</box>", re.S)


@dataclass
class Rollout:
    answer: str
    boxes: np.ndarray  # [M, 4]
    spans: list[tuple[int, int]]  # token span of each box within the answer
    forced: bool = False  # injected rather than sampled; see `refusal_rollout`
    credits: torch.Tensor | None = None
    reward: float = 0.0
    metrics: dict = field(default_factory=dict)
    action_latents: torch.Tensor | None = None
    action_log_probs: torch.Tensor | None = None
    action_point_mask: torch.Tensor | None = None
    old_log_probs: torch.Tensor | None = None
    reference_log_probs: torch.Tensor | None = None
    sampling_temperature: float | None = None  # forced refusal has no sampling temperature
    # Joint path: the sampled contours' sequence reward (``reward`` prices the
    # deterministic ones), and each box's sampled-minus-deterministic credit.
    sampled_reward: float = 0.0
    action_deltas: torch.Tensor | None = None
    action_active: torch.Tensor | None = None
    # Detached backbone read-outs from collection, so an action-only replay
    # can rebuild the contour graph without another backbone pass.
    replay_context: dict | None = None


class ReferenceSnapshot:
    """Frozen copies of the trainable backbone tensors at the start of a stage."""

    def __init__(self, parameters, values=None):
        self.parameters = list(parameters)
        if values is None:
            values = [p.detach().clone() for p in self.parameters]
        self.values = [v.to(p.device, p.dtype) for p, v in zip(self.parameters, values)]
        if any(v.shape != p.shape for p, v in zip(self.parameters, self.values)):
            raise ValueError("reference snapshot does not match the trainable tensors")


@contextmanager
def as_reference(model):
    """Run as the frozen reference.

    With a ``ReferenceSnapshot`` on ``model.rl_reference`` the trainable
    backbone tensors are swapped for their stage-start values. The swap is an
    in-place copy, so it must not happen while a graph that saved those tensors
    is waiting for its backward pass - callers compute the reference first.

    Without one, LoRA is switched off: LoRA is additive with a scalar gain, so
    the pre-training model is the same weights with that gain at zero.
    """
    snapshot = getattr(model, "rl_reference", None)
    if snapshot is not None:
        with torch.no_grad():
            live = [p.detach().clone() for p in snapshot.parameters]
            for parameter, value in zip(snapshot.parameters, snapshot.values):
                parameter.copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                for parameter, value in zip(snapshot.parameters, live):
                    parameter.copy_(value)
        return
    layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    saved = [m.scaling for m in layers]
    for module in layers:
        module.scaling = 0.0
    try:
        yield
    finally:
        for module, scale in zip(layers, saved):
            module.scaling = scale


def box_token_spans(tokenizer, answer: str, special=None) -> list[tuple[int, int]]:
    """Token span of every ``<box>...</box>``, as offsets into the answer.

    Measured by tokenising prefixes rather than by aligning tokens to
    characters: box coordinates are single special tokens, so the boundaries
    fall exactly where the regex does, and prefix lengths are the cheapest way
    to say so without depending on a fast tokeniser's offset mapping.
    """
    if special is not None:
        ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        return [(int(s[0]), int(s[-1]) + 1) for s in find_box_token_spans(ids, special)]
    spans = []
    for match in BOX.finditer(answer):
        start = len(tokenizer(answer[: match.start()], add_special_tokens=False)["input_ids"])
        end = len(tokenizer(answer[: match.end()], add_special_tokens=False)["input_ids"])
        if end > start:
            spans.append((start, end))
    return spans


def sample_group(
    model,
    image,
    expression: str,
    group: int,
    temperature: float,
    top_p: float,
    temperature_jitter: float = 0.0,
) -> list[Rollout]:
    """Sample answers with one uniform temperature draw per rollout.

    Each answer uses a fixed temperature in [temperature-jitter, temperature+jitter].
    Jitter draws follow the seeded PyTorch CPU RNG. Zero jitter preserves the old
    fixed-temperature behavior without consuming extra random numbers. This only
    changes generation, not the existing untempered replay likelihood convention.
    """
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(temperature_jitter) or temperature_jitter < 0:
        raise ValueError("temperature_jitter must be finite and nonnegative")
    if temperature_jitter > 0 and (
        temperature <= temperature_jitter or not math.isfinite(temperature + temperature_jitter)
    ):
        raise ValueError("temperature jitter interval must be strictly positive and finite")
    tokenizer = model.backbone.tokenizer
    rollouts = []
    for _ in range(group):
        sampled_temperature = temperature
        if temperature_jitter > 0:
            sampled_temperature += temperature_jitter * (
                2.0 * torch.rand((), device="cpu", dtype=torch.float64).item() - 1.0
            )
        result = model.predict(
            image, expression, contours=False,
            temperature=sampled_temperature, top_p=top_p,
        )
        answer = result["answer"]
        rollouts.append(
            Rollout(
                answer=answer,
                boxes=np.asarray(result["boxes"], dtype=np.float32).reshape(-1, 4),
                spans=box_token_spans(tokenizer, answer, model.backbone.special),
                sampling_temperature=sampled_temperature,
            )
        )
    return rollouts


def refusal_rollout(model, template: str = "") -> Rollout:
    """A refusal, inserted into the group rather than waited for.

    RC-GRPO's finding, and it applies to us unchanged: on a no-target sample the
    policy essentially never samples a refusal, so every rollout in the group
    scores the same zero, the advantage vanishes and nothing is learned. Putting
    one in by hand makes the comparison well-defined. It is off-policy, and the
    bias shrinks by itself as the policy starts producing refusals on its own.

    ``NONE_TOKEN`` is the refusal string and is asserted against the tokenizer
    at load, so the target is a token the decoder actually emits.
    """
    answer = template or f"<box>{NONE_TOKEN}</box>"
    return Rollout(answer=answer, boxes=np.zeros((0, 4), np.float32), spans=[], forced=True)


def answer_labels(tokenizer, prompt: str, answer: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(input_ids, labels)`` for a teacher-forced pass over one rollout.

    Only the answer is supervised; the prompt carries the image and is masked
    out, so the log-probs come back aligned with the answer's own tokens and the
    spans from ``box_token_spans`` index straight into them.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prompt_ids + answer_ids], device=device)
    labels = torch.full_like(input_ids, -100)
    labels[0, len(prompt_ids):] = torch.tensor(answer_ids, device=device)
    return input_ids, labels
