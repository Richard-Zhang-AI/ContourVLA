"""The vision-language interaction module of ContourVLA.

The embedded vision-language model maps an image and a referring
expression to box tokens. What
this module adds is a second, non-generative read-out of the same forward pass,
which is what the contour policy consumes:

* ``instance_features`` - hidden states pooled over each ``<box> ... </box>``
  run, i.e. one action-ready vector per detected object. These positions follow
  the referring expression, so this is the branch's only language-conditioned
  input: it is what says *which* object a contour belongs to;
* ``visual_grid`` - hidden states at the ``<IMG_CONTEXT>`` positions folded back
  into a 2-D map. These positions *precede* the expression, so under a causal
  mask they are a function of the image alone, at every depth. The grid supplies
  appearance and boundary evidence, not instruction selectivity.

The stock ``Qwen2Model.forward`` builds a block-diffusion attention mask for the
base model's parallel box decoding, which only makes sense for its packed MTP
training format. Teacher forcing a plain ``[prompt, answer]`` sequence needs an
ordinary causal mask, so the decoder stack is driven directly here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import os
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import BackboneConfig
from ..tokens import SpecialTokenIds
from .lora import inject_lora

DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


@dataclass
class BackboneOutput:
    hidden_states: torch.Tensor  # [B, L, D_llm]
    instance_features: torch.Tensor  # [M, D_llm], M = total boxes in the batch
    instance_batch_index: torch.Tensor  # [M]
    visual_grid: list[torch.Tensor]  # B x [D_llm, h, w]
    grid_sizes: torch.Tensor  # [B, 2] merged (h, w)
    detection_loss: torch.Tensor | None
    num_supervised_tokens: int
    # Every layer that was tapped, for consumers that pick at run time rather
    # than from config - currently only the layer router.
    tapped: dict[int, torch.Tensor] = field(default_factory=dict)


def load_checkpoint_payload(path: str) -> dict[str, Any]:
    """Load the single public checkpoint used to build the model."""
    if not path:
        raise ValueError("backbone.checkpoint_path must point to contourvla.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"invalid ContourVLA checkpoint: {path}")
    if "backbone_bundle" not in payload:
        raise ValueError(
            "contourvla.pt does not contain the embedded model runtime; "
            "use the released standalone checkpoint"
        )
    return payload


def materialize_runtime_bundle(payload: dict[str, Any]) -> str:
    """Unpack the small architecture/tokenizer bundle embedded in contourvla.pt."""
    root = Path(tempfile.mkdtemp(prefix="contourvla_runtime_"))
    bundle = payload["backbone_bundle"]
    if not isinstance(bundle, dict) or not bundle:
        raise ValueError("checkpoint runtime bundle is empty")
    for name, data in bundle.items():
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid runtime bundle path: {name!r}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return str(root)


class VLInteractionModule(nn.Module):
    def __init__(
        self, config: BackboneConfig, payload: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

        self.config = config
        self.dtype_ = DTYPES[config.torch_dtype]
        if payload is None:
            payload = load_checkpoint_payload(config.checkpoint_path)
        self._runtime_dir = materialize_runtime_bundle(payload)

        hf_config = AutoConfig.from_pretrained(
            self._runtime_dir, trust_remote_code=True, local_files_only=True
        )
        # The released config asks for MagiAttention (Hopper-only kernel). SDPA is
        # the portable path and is the only one that accepts the explicit 4-D mask
        # this wrapper builds.
        hf_config._attn_implementation = config.attn_implementation
        hf_config.text_config._attn_implementation = config.attn_implementation
        hf_config.vision_config._attn_implementation = config.attn_implementation
        hf_config.text_config.use_cache = False

        self.model = AutoModel.from_config(hf_config, trust_remote_code=True)
        self.model.to(dtype=self.dtype_)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._runtime_dir,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self._runtime_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.processor.image_processor.in_token_limit = config.in_token_limit
        self.special = SpecialTokenIds.from_model(hf_config, self.tokenizer)
        self.hidden_size = int(hf_config.text_config.hidden_size)
        self.extra_taps: set[int] = set()
        self.merge_size = int(hf_config.vision_config.merge_kernel_size[0])
        self.patch_size = int(hf_config.vision_config.patch_size)

        self._configure_trainable()
    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #

    def _configure_trainable(self) -> None:
        cfg = self.config
        language_model = self.model.language_model

        if cfg.freeze_vision:
            self.model.vision_model.requires_grad_(False)
        if cfg.freeze_projector:
            self.model.mlp1.requires_grad_(False)

        if cfg.llm_lora_rank > 0:
            language_model.requires_grad_(False)
            self.num_lora_modules = inject_lora(
                language_model.model,
                rank=cfg.llm_lora_rank,
                alpha=cfg.llm_lora_alpha,
                dropout=cfg.llm_lora_dropout,
            )
        else:
            self.num_lora_modules = 0
            language_model.requires_grad_(not cfg.freeze_llm)
            # `lm_head` is tied to the input embeddings; training it while the
            # rest of the LLM is frozen would silently unfreeze the embeddings.
            if cfg.freeze_llm:
                language_model.lm_head.requires_grad_(False)

    @property
    def llm(self):
        return self.model.language_model

    @property
    def decoder_layers(self) -> nn.ModuleList:
        return self.model.language_model.model.layers

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #

    def encode_vision(
        self, pixel_values: torch.Tensor, image_grid_hws: torch.Tensor
    ) -> torch.Tensor:
        """Patches -> projected visual tokens, ``[total_tokens, D_llm]``."""
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype_)
        image_grid_hws = image_grid_hws.to(device=self.device, dtype=torch.int32)
        embeds = self.model.extract_feature(pixel_values, image_grid_hws)
        if isinstance(embeds, (list, tuple)):
            embeds = torch.cat(list(embeds), dim=0)
        return self.model.mlp1(embeds)

    def _causal_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch, length = attention_mask.shape
        minimum = torch.finfo(dtype).min
        causal = torch.full((length, length), minimum, device=attention_mask.device, dtype=dtype)
        causal.triu_(diagonal=1)
        mask = causal[None, None].expand(batch, 1, length, length).clone()
        padding = (attention_mask == 0)[:, None, None, :]
        return mask.masked_fill(padding, minimum)

    def _resolve_layer(self, value: int, name: str) -> int:
        count = len(self.decoder_layers)
        index = int(value)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise ValueError(f"{name} {value} is outside the {count} decoder layers")
        return index

    def _run_decoder(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Final hidden states, plus every layer the contour branch taps.

        Detection has to read the last layer - that is what ``lm_head``
        projects - but the contour branch does not, and its two inputs do not
        even want the same depth. See ``BackboneConfig`` for why.
        """
        base = self.llm.model
        mask = self._causal_mask(attention_mask, inputs_embeds.dtype)
        hidden = inputs_embeds
        wanted = {
            self._resolve_layer(self.config.instance_layer, "instance_layer"),
            self._resolve_layer(self.config.grid_layer, "grid_layer"),
        }
        # The router asks for a spread of layers on top of the two the branch
        # has always read. Set by the owner rather than by config so the
        # backbone does not have to know the router exists.
        wanted |= getattr(self, "extra_taps", set())
        tapped: dict[int, torch.Tensor] = {}
        use_checkpoint = self.config.gradient_checkpointing and self.training
        for depth, layer in enumerate(self.decoder_layers):
            if use_checkpoint:
                hidden = checkpoint(
                    lambda h, m, p, _layer=layer: _layer(
                        h, attention_mask=m, position_ids=p, past_key_value=None,
                        output_attentions=False, use_cache=False,
                    )[0],
                    hidden,
                    mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden = layer(
                    hidden,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )[0]
            if depth in wanted:
                tapped[depth] = hidden
        final = base.norm(hidden)
        # The last layer is normalised before use, so tapping it must give the
        # same tensor the detection head sees. Earlier layers are handed over
        # raw; every consumer normalises its own input.
        tapped[len(self.decoder_layers) - 1] = final
        return final, tapped

    def _detection_loss(
        self, hidden: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor | None, int]:
        """Cross-entropy over the answer tokens only.

        The logits are materialised for the supervised positions alone. With a
        152k vocabulary and sequences of a few thousand tokens, projecting the
        whole sequence would dominate both memory and compute for the sake of a
        few dozen supervised positions.
        """
        source = hidden[:, :-1]
        target = labels[:, 1:]
        selected = target != -100
        if not bool(selected.any()):
            return None, 0
        logits = self.llm.lm_head(source[selected])
        loss = F.cross_entropy(logits.float(), target[selected])
        return loss, int(selected.sum())

    def answer_log_probs(
        self, hidden: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """log pi(taken token) at each supervised position: ``[T]``.

        What a policy gradient needs, and it reuses ``_detection_loss``'s trick
        of projecting only the supervised positions - with a 152k vocabulary and
        a few thousand tokens of context, materialising the whole sequence's
        logits would dominate memory for the sake of a few dozen answer tokens.

        Teacher-forced over an already-generated sequence rather than captured
        during sampling: the vendored decoder does not expose its distributions,
        and one extra prefill is far simpler than threading them out of it.
        """
        source = hidden[:, :-1]
        target = labels[:, 1:]
        selected = target != -100
        if not bool(selected.any()):
            return hidden.new_zeros(0)
        logits = self.llm.lm_head(source[selected]).float()
        return torch.log_softmax(logits, dim=-1).gather(
            1, target[selected][:, None]
        ).squeeze(1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        box_token_index: torch.Tensor,
        box_token_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> BackboneOutput:
        """
        Args:
            pixel_values: ``[total_patches, 3, P, P]`` for the whole batch.
            image_grid_hws: ``[B, 2]`` pre-merge patch grid per image.
            input_ids / attention_mask: right-padded ``[B, L]``.
            box_token_index: ``[B, N_max, 6]`` positions of each box's tokens.
            box_token_mask: ``[B, N_max]`` validity of each box slot.
            labels: ``[B, L]`` with ``-100`` everywhere but the answer tokens.
        """
        device = self.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        visual_tokens = self.encode_vision(pixel_values, image_grid_hws)
        embeds = self.llm.get_input_embeddings()(input_ids)

        batch, length, dim = embeds.shape
        flat_ids = input_ids.reshape(-1)
        flat_embeds = embeds.reshape(-1, dim)
        image_slots = flat_ids == self.special.image_context
        num_slots = int(image_slots.sum())
        if num_slots != visual_tokens.shape[0]:
            raise ValueError(
                f"{num_slots} <IMG_CONTEXT> placeholders but {visual_tokens.shape[0]} visual tokens"
            )
        # Out-of-place so autograd reaches the vision tower and the projector.
        flat_embeds = flat_embeds.masked_scatter(
            image_slots[:, None], visual_tokens.to(flat_embeds.dtype)
        )
        embeds = flat_embeds.reshape(batch, length, dim)

        position_ids = (
            torch.arange(length, device=device).unsqueeze(0).expand(batch, -1)
        )
        hidden, tapped = self._run_decoder(embeds, attention_mask, position_ids)

        detection_loss, num_tokens = (None, 0)
        if labels is not None:
            detection_loss, num_tokens = self._detection_loss(hidden, labels.to(device))

        instance_features, instance_batch_index = self.pool_box_features(
            tapped[self._resolve_layer(self.config.instance_layer, "instance_layer")],
            box_token_index.to(device),
            box_token_mask.to(device),
        )
        visual_grid, grid_sizes = self.fold_visual_tokens(
            tapped[self._resolve_layer(self.config.grid_layer, "grid_layer")],
            input_ids,
            image_grid_hws,
        )

        return BackboneOutput(
            hidden_states=hidden,
            instance_features=instance_features,
            instance_batch_index=instance_batch_index,
            visual_grid=visual_grid,
            grid_sizes=grid_sizes,
            detection_loss=detection_loss,
            num_supervised_tokens=num_tokens,
            tapped=tapped,
        )

    # ------------------------------------------------------------------ #
    # read-outs
    # ------------------------------------------------------------------ #

    @staticmethod
    def pool_box_features(
        hidden: torch.Tensor, box_token_index: torch.Tensor, box_token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One feature vector per box, flattened across the batch.

        A box always occupies six positions, so gathering them and averaging
        gives a fixed-size descriptor regardless of how many objects an image
        contains. The batch is flattened into ``M`` instances here; that flat
        layout is what makes the downstream decoder indifferent to the varying
        number of detections per image.
        """
        batch, num_boxes, span = box_token_index.shape
        dim = hidden.shape[-1]
        index = box_token_index.clamp(min=0).reshape(batch, num_boxes * span, 1)
        gathered = hidden.gather(1, index.expand(-1, -1, dim))
        gathered = gathered.reshape(batch, num_boxes, span, dim).mean(dim=2)

        valid = box_token_mask.bool()
        instance_features = gathered[valid]
        batch_index = (
            torch.arange(batch, device=hidden.device).unsqueeze(1).expand(-1, num_boxes)[valid]
        )
        return instance_features, batch_index

    def fold_visual_tokens(
        self, hidden: torch.Tensor, input_ids: torch.Tensor, image_grid_hws: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """``<IMG_CONTEXT>`` hidden states back into per-image 2-D maps."""
        grids: list[torch.Tensor] = []
        sizes = []
        merged = image_grid_hws.to(torch.int64) // self.merge_size
        for b in range(hidden.shape[0]):
            slots = input_ids[b] == self.special.image_context
            tokens = hidden[b][slots]
            h, w = int(merged[b, 0]), int(merged[b, 1])
            if tokens.shape[0] != h * w:
                raise ValueError(
                    f"image {b}: {tokens.shape[0]} visual tokens for a {h}x{w} grid"
                )
            grids.append(tokens.reshape(h, w, -1).permute(2, 0, 1).contiguous())
            sizes.append((h, w))
        return grids, torch.tensor(sizes, device=hidden.device, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_answer(
        self,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        generation_mode: str = "hybrid",
        **kwargs: Any,
    ) -> str:
        """Autoregressive box decoding using the released generation loop."""
        return self.model.generate(
            pixel_values=pixel_values.to(self.device, self.dtype_),
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            image_grid_hws=image_grid_hws.to(self.device, torch.int32),
            tokenizer=self.tokenizer,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            generation_mode=generation_mode,
            **kwargs,
        )

    @property
    def num_layers(self) -> int:
        return len(self.decoder_layers)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
