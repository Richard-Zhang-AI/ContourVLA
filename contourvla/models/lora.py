"""Minimal in-place LoRA injection.

`peft` would work too, but it re-parents the model into a `PeftModel`, and the
backbone wrapper walks the Qwen2 module tree directly to run its own causal
forward. Replacing `nn.Linear` in place keeps that tree untouched.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

DEFAULT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        dtype = base.weight.dtype
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        update = self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()
        return out + update * self.scaling

    @property
    def weight(self) -> torch.Tensor:  # keeps dtype/device checks working
        return self.base.weight


def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> int:
    """Swap every targeted ``nn.Linear`` for a LoRA-augmented copy."""
    if rank <= 0:
        return 0
    targets = tuple(targets)
    replaced = 0
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
                replaced += 1
    return replaced


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in module.named_parameters() if "lora_a" in n or "lora_b" in n]
