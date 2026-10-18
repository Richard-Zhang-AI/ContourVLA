"""Routing multimodal context out of the language model, one layer per block.

The contour branch reads two hand-picked layers: 18 for the spatial grid and
the last for the box descriptor. The default taps are fixed in the config, while MoRoute (arXiv 2607.29545)
identifies as leaving a hierarchy underused - it lets every DiT block learn
which VLM layer it wants instead. This is that idea, with two changes.

**What is routed.** MoRoute routes the layer's whole condition sequence. The
contour branch's context is mostly image tokens, and attending to all of them
from every contour point is expensive. This module routes the *text*
positions instead. A referring expression is short, and under causal masking
an image token never sees the expression that follows it, so no layer of the
image grid is language-conditioned. Text tokens sit after the image and have
attended over it, so they carry that conditioning.

**Where it is injected.** Into the action decoder's existing cross-attention
memory, per decoder layer, which means the decoder's layers can no longer share
one memory tensor and are unrolled in ``ChunkDecoderBlock``.

The output projection is zero-initialised, so with the router switched on the
model is bit-identical to the model without it until training moves it. That
gate lives on the consumer side - the cross-attention in ``ChunkDecoderBlock``
that this feeds - and only there: putting a second one on this module's own
output deadlocks the pair, since each zero blocks the gradient the other needs
to move. That is a guarantee about the starting point and not about the end.

---

**State conditioning (this revision).** The original router picks a layer from
two things only: which (decoder layer, call) slot is asking - a learned
``query`` row, i.e. a fixed identity that never looks at the sample - and which
image is being described - the pooled text keys. Nothing about the *contour*
being evolved (its size, shape, how irregular it already is) or the *evolution*
it is in the middle of (which call, how far the previous call actually moved it)
reaches the routing decision at all. That is precisely the gap the VLA framing
warns about: a policy conditioned on the wrong slice of the state is a policy
that cannot use the rest of it, however expressive its action head is.

``select`` now optionally takes ``state_features``, a small per-*instance*
(not per-image) vector built by ``models.action_decoder.evolution_state_features``
from the box, the current contour, and - from the second call on - how far the
previous call actually moved it. It is folded into the query before the
layer comparison, so two objects in the same image, or the same object at its
two different calls, can land on different candidate layers when their state
warrants it; previously they could not, because nothing distinguished them
besides the frozen per-slot identity.

The mechanism is engineered to be provably inert until it has a reason not to
be, the same way the rest of this module is: the projection that turns
``state_features`` into a query offset ends in a zero-initialised linear layer,
so at initialisation - and after loading a checkpoint that predates this field,
where the tensor is simply absent - its output is exactly zero for every input,
and the query is exactly what it was before. Only training can move it off
zero, and only once the surrounding zero-initialised consumer (``out_proj`` in
``ChunkDecoderBlock``) has already started passing gradient back through here -
so the state signal is available from the same step the routed branch is,
never before.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# Size of the per-instance state vector ``evolution_state_features`` builds:
# log aspect ratio, log scale, point-budget ratio, contour irregularity,
# call progress, mean previous-call displacement, max previous-call
# displacement. Kept here rather than inferred so the router and the feature
# builder cannot silently drift apart.
STATE_DIM = 7


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
    )


class LayerRouter(nn.Module):
    """Picks a language-model layer for each decoder layer, per call.

    ``slots`` are (decoder layer, call) pairs rather than decoder layers alone.
    The two calls are separate stages of the same evolution - the first moves
    the contour most of the way, the second refines it - which is the same
    argument MoRoute makes for routing by denoising step, and it doubles the
    routing decisions on a decoder that only has six layers.
    """

    def __init__(
        self,
        candidates: tuple[int, ...],
        llm_dim: int,
        dim: int,
        slots: int,
        key_dim: int = 64,
        max_tokens: int = 32,
        prior_scale: float = 4.0,
        state_dim: int = STATE_DIM,
        state_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.candidates = tuple(candidates)
        self.slots = slots
        self.max_tokens = max_tokens
        self.prior_scale = prior_scale

        self.layer_identity = nn.Parameter(torch.zeros(len(self.candidates), llm_dim))
        nn.init.normal_(self.layer_identity, std=0.02)
        self.key_mlp = _mlp(llm_dim, key_dim * 2, key_dim)
        self.query = nn.Parameter(torch.randn(slots, key_dim) * 0.02)

        # One projection per slot, unshared: each decoder layer should be able
        # to adapt its chosen layer its own way.
        #
        # NOT zero-initialised, though the no-op guarantee is what motivated it
        # here. The consumer already zeroes the output projection of the
        # cross-attention this feeds (see ChunkDecoderBlock), and the two sit in
        # series, so zeroing both is a deadlock rather than two safety nets:
        # d(loss)/d(out_proj.W) is the incoming gradient times *this* module's
        # output, which is zero, and the gradient back to here passes through
        # out_proj.W, which is also zero. Neither can be the first to move, and
        # the branch stays dead for the whole run. One gate is enough, and it
        # belongs at the consumer, which is where the residual is added.
        self.projection = nn.ModuleList(
            [_mlp(llm_dim, dim, dim) for _ in range(slots)]
        )

        # Per-instance contour/evolution state, folded into the query before it
        # meets the keys. Zero-initialised at the last linear layer for the
        # same reason the router does not need a second zero elsewhere: this
        # sits in a *sum* with ``query`` (not in series with a downstream
        # zero), so a zero here only pins the *offset* at zero, not the
        # gradient path - ``query`` keeps training even before this module
        # wakes up, and this module gets gradient through the sum from the
        # first step, unlike the out_proj/router-projection pair above.
        self.state_conditioning = state_conditioning
        self.state_mlp = _mlp(state_dim, key_dim * 2, key_dim) if state_conditioning else None
        if self.state_mlp is not None:
            nn.init.zeros_(self.state_mlp[-1].weight)
            nn.init.zeros_(self.state_mlp[-1].bias)

        self.register_buffer("prior", self._gaussian_prior(), persistent=False)

    def _gaussian_prior(self) -> torch.Tensor:
        """Near-diagonal bias mapping slot depth to layer depth.

        Without it the router has no notion that either axis is ordered, and
        MoRoute reports early training either unstable or collapsed. Annealed
        away over training so the learned logits take over.
        """
        slots = torch.arange(self.slots, dtype=torch.float32) / max(self.slots - 1, 1)
        layers = torch.arange(len(self.candidates), dtype=torch.float32) / max(
            len(self.candidates) - 1, 1
        )
        distance = slots[:, None] - layers[None, :]
        return torch.exp(-0.5 * (distance / 0.25) ** 2)

    def text_context(
        self,
        tapped: dict[int, torch.Tensor],
        text_mask: torch.Tensor,
        expression_span: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``([C, B, T, D], [B, T])`` hidden states at routed positions, padded.

        With ``expression_span`` this is the referring expression and nothing
        else. Without it, the last ``max_tokens`` text positions are used. The
        template prefix is longer than a short expression, so that window is
        mostly boilerplate.
        """
        batch, length = text_mask.shape
        device = text_mask.device
        order = torch.arange(length, device=device)[None, :].expand(batch, -1)
        if expression_span is not None:
            span = expression_span.to(device)
            keep = (order >= span[:, :1]) & (order < span[:, 1:2]) & text_mask
            # An empty span would silently route nothing; fall back rather than
            # hand the decoder a zero-width memory it cannot attend over.
            empty = ~keep.any(dim=1)
            if bool(empty.any()):
                rank = text_mask.flip(1).cumsum(1).flip(1)
                window = text_mask & (rank <= self.max_tokens)
                keep = torch.where(empty[:, None], window, keep)
        else:
            # Rank text positions from the end; keep the last max_tokens of them.
            rank = text_mask.flip(1).cumsum(1).flip(1)
            keep = text_mask & (rank <= self.max_tokens)

        counts = keep.sum(1)
        width = int(counts.max().clamp(min=1))
        index = torch.zeros(batch, width, dtype=torch.long, device=device)
        valid = torch.zeros(batch, width, dtype=torch.bool, device=device)
        for row in range(batch):
            positions = order[row][keep[row]]
            index[row, : positions.numel()] = positions
            valid[row, : positions.numel()] = True

        stacked = []
        for layer in self.candidates:
            # float32 throughout: the backbone runs in bfloat16 and the router's
            # own parameters do not, and a LayerNorm will refuse the mismatch
            # rather than cast for you. The router is small enough that the
            # precision costs nothing.
            hidden = tapped[layer].float()
            stacked.append(hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1])))
        return torch.stack(stacked), valid

    def keys(self, context: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``[C, B, key_dim]`` normalised per-image keys, one per candidate layer.

        Split out of ``forward`` so a caller that issues several ``select``
        calls per sample - one per call of the action decoder, so the routing
        decision can see how the contour changed in between - pools and
        projects the (image-level, call-independent) text context exactly
        once rather than once per call.

        Keys pool exactly what the values are read from. Pooling over every
        text position mixes in the template and leaves little that differs
        across images.
        """
        weights = valid.float()[None, ..., None]
        pooled = (context * weights).sum(2) / weights.sum(2).clamp(min=1)  # [C, B, D]
        keys = self.key_mlp(pooled + self.layer_identity[:, None, :])
        return torch.nn.functional.normalize(keys, dim=-1)

    def select(
        self,
        context: torch.Tensor,
        valid: torch.Tensor,
        keys: torch.Tensor,
        batch_index: torch.Tensor,
        slot_offset: int,
        num_slots: int,
        anneal: float = 1.0,
        state_features: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """``(per-slot memory [M, T, dim], padding [M, T], routing stats)``.

        ``context``/``valid``/``keys`` come from ``text_context``/``keys`` and
        are image-level; ``batch_index`` gathers them onto the ``M`` flattened
        instances the action decoder actually works on, which is also the
        granularity ``state_features`` (``[M, state_dim]``, optional) is given
        at. This is what lets two instances that share an image, or one
        instance across its two calls, pick different candidate layers: with
        ``state_features=None`` every instance of an image is handed the same
        query, exactly reproducing the previous per-image behaviour.

        ``slot_offset``/``num_slots`` select a contiguous run of this router's
        (decoder layer, call) slots, so the action decoder can call this once
        per call with only that call's ``num_layers`` slots and its own
        ``state_features``, instead of committing to every call's routing
        before the first one has actually moved the contour.
        """
        candidates = context.shape[0]
        slot_slice = slice(slot_offset, slot_offset + num_slots)
        # Raw, not normalised: normalising once *before* an optional state
        # offset and again *after* is a no-op in real arithmetic when the
        # offset is exactly zero, but not in float32 - the second normalise's
        # own division rounds differently from the first's, so "the offset is
        # zero at initialisation" would only make the output close to what it
        # was, not equal to it. Each branch below normalises exactly once, so
        # the state-conditioned path at a zero offset is bit-identical to the
        # unconditioned one rather than merely near it.
        raw_query = self.query[slot_slice].float()  # [S, K]
        prior = self.prior[slot_slice]  # [S, C]

        keys_per_instance = keys[:, batch_index]  # [C, M, K]
        context_per_instance = context[:, batch_index]  # [C, M, T, D]
        valid_per_instance = valid[batch_index]  # [M, T]

        if self.state_mlp is not None and state_features is not None:
            offset = self.state_mlp(state_features.float())[:, None, :]  # [M, 1, K]
            query = torch.nn.functional.normalize(raw_query[None, :, :] + offset, dim=-1)  # [M,S,K]
            logits = torch.einsum("msk,cmk->msc", query, keys_per_instance)
        else:
            query = torch.nn.functional.normalize(raw_query, dim=-1)  # [S, K]
            logits = torch.einsum("sk,cmk->msc", query, keys_per_instance)
        logits = logits + anneal * self.prior_scale * prior[None]

        probabilities = logits.softmax(dim=-1)  # [M, S, C]
        hard = torch.nn.functional.one_hot(
            probabilities.argmax(dim=-1), candidates
        ).to(probabilities.dtype)
        # Straight-through: forward the discrete pick, differentiate the soft one.
        selection = hard + probabilities - probabilities.detach()

        memories = []
        for local_slot in range(num_slots):
            weight = selection[:, local_slot]  # [M, C]
            chosen = torch.einsum("cmtd,mc->mtd", context_per_instance, weight)
            memories.append(self.projection[slot_offset + local_slot](chosen))

        return memories, ~valid_per_instance, {
            "probabilities": probabilities,
            "selection": probabilities.argmax(dim=-1),
        }

    def forward(
        self,
        tapped: dict[int, torch.Tensor],
        text_mask: torch.Tensor,
        batch_index: torch.Tensor,
        anneal: float = 1.0,
        expression_span: torch.Tensor | None = None,
        state_features: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Single-shot convenience path: every slot at once, image-level unless
        ``state_features`` is given. ``ContourVLA.route`` uses this. Training
        and inference call ``text_context`` / ``keys`` / ``select`` directly,
        once per call, so later calls can condition on how the contour actually
        moved in the calls before them.
        """
        context, valid = self.text_context(tapped, text_mask, expression_span)
        keys = self.keys(context, valid)
        return self.select(
            context, valid, keys, batch_index, 0, self.slots, anneal, state_features
        )


def routing_losses(
    probabilities: torch.Tensor,
    diversity: float = 0.01,
    confidence: float = 0.01,
    monotonic: float = 0.001,
) -> torch.Tensor:
    """The three terms that stop routing degenerating.

    MoRoute reports all three as necessary: without diversity the whole model
    collapses onto a couple of layers, without confidence the selection stays
    soft and the straight-through estimator is fighting a near-uniform
    distribution, and without monotonicity there is nothing tying deeper
    decoder layers to deeper language-model layers.
    """
    usage = probabilities.mean(dim=(0, 1))  # [C]
    spread = -(usage * (usage + 1e-9).log()).sum()  # want this large
    sharp = -(probabilities.max(dim=-1).values.mean())  # want the peak large

    slots = probabilities.shape[1]
    position = torch.arange(probabilities.shape[-1], device=probabilities.device).float()
    centre = (probabilities * position).sum(-1)  # [B, slots]
    order = centre[:, 1:] - centre[:, :-1] if slots > 1 else centre.new_zeros(1)
    backwards = torch.relu(-order).mean()

    return -diversity * spread + confidence * sharp + monotonic * backwards
