"""Prompt/answer serialisation and box-token bookkeeping for the base VLM.

The base VLM emits boxes as literal vocabulary tokens rather than through a
regression head::

    <ref>a red cup</ref><box><x1><y1><x2><y2></box><box><x1><y1><x2><y2></box>

with each coordinate an integer in ``[0, 1000]`` mapped to the token ``<k>``.
A box therefore always occupies exactly six consecutive positions, which is what
lets us pull one "action-ready" feature vector per detected instance straight
out of the language model hidden states.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

COORD_BINS = 1000
TOKENS_PER_BOX = 6  # <box> x1 y1 x2 y2 </box>

IMAGE_START_TOKEN = "<img>"
IMAGE_END_TOKEN = "</img>"
IMAGE_CONTEXT_TOKEN = "<IMG_CONTEXT>"
BOX_START_TOKEN = "<box>"
BOX_END_TOKEN = "</box>"
REF_START_TOKEN = "<ref>"
REF_END_TOKEN = "</ref>"
# Capitalised because the base model recognises an empty box by the token
# "None". Lowercase "none" is a different token. ``SpecialTokenIds.from_model``
# asserts the string and the id agree.
NONE_TOKEN = "None"

SYSTEM_PROMPT = "You are a helpful assistant."

BOX_PATTERN = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
# Built from the constants rather than spelled out, so it cannot drift away
# from NONE_TOKEN.
EMPTY_BOX_PATTERN = re.compile(
    re.escape(f"{BOX_START_TOKEN}{NONE_TOKEN}{BOX_END_TOKEN}")
)


@dataclass(frozen=True)
class SpecialTokenIds:
    image_context: int
    box_start: int
    box_end: int
    ref_start: int
    ref_end: int
    coord_start: int
    coord_end: int
    none: int
    im_end: int
    pad: int

    @classmethod
    def from_model(cls, config, tokenizer) -> "SpecialTokenIds":
        def tid(token: str) -> int:
            value = tokenizer.convert_tokens_to_ids(token)
            if value is None or value < 0:
                raise ValueError(f"tokenizer is missing the special token {token!r}")
            return int(value)

        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tid("<|endoftext|>")

        # ``build_answer`` writes NONE_TOKEN as text and the model's decoder
        # recognises an empty box by comparing against ``none_token_id``. If the
        # string does not tokenise to exactly that id the two never meet, and
        # training optimises a token the decoder ignores.
        declared = int(getattr(config, "none_token_id", tid(NONE_TOKEN)))
        encoded = tokenizer(NONE_TOKEN, add_special_tokens=False)["input_ids"]
        if encoded != [declared]:
            raise ValueError(
                f"NONE_TOKEN {NONE_TOKEN!r} encodes to {encoded} but the model "
                f"declares none_token_id={declared} "
                f"({tokenizer.convert_ids_to_tokens([declared])}). The answer "
                f"string and the decoder must agree on this token."
            )

        return cls(
            image_context=int(getattr(config, "image_token_index", tid(IMAGE_CONTEXT_TOKEN))),
            box_start=int(getattr(config, "box_start_token_id", tid(BOX_START_TOKEN))),
            box_end=int(getattr(config, "box_end_token_id", tid(BOX_END_TOKEN))),
            ref_start=int(getattr(config, "ref_start_token_id", tid(REF_START_TOKEN))),
            ref_end=int(getattr(config, "ref_end_token_id", tid(REF_END_TOKEN))),
            coord_start=int(getattr(config, "coord_start_token_id", tid("<0>"))),
            coord_end=int(getattr(config, "coord_end_token_id", tid(f"<{COORD_BINS}>"))),
            none=declared,
            im_end=tid("<|im_end|>"),
            pad=int(pad),
        )

    def coord_token_id(self, value: int) -> int:
        return self.coord_start + int(np.clip(value, 0, COORD_BINS))

    def is_coord(self, token_id: int) -> bool:
        return self.coord_start <= token_id <= self.coord_end


def quantize_box(box: Sequence[float]) -> tuple[int, int, int, int]:
    """Normalised ``(x1, y1, x2, y2)`` in ``[0, 1]`` to integer coordinate bins."""
    x1, y1, x2, y2 = (float(np.clip(v, 0.0, 1.0)) for v in box)
    return tuple(int(round(v * COORD_BINS)) for v in (x1, y1, x2, y2))  # type: ignore[return-value]


def dequantize_box(bins: Sequence[int]) -> tuple[float, float, float, float]:
    return tuple(float(np.clip(v, 0, COORD_BINS)) / COORD_BINS for v in bins)  # type: ignore[return-value]


def format_box(box: Sequence[float]) -> str:
    x1, y1, x2, y2 = quantize_box(box)
    return f"{BOX_START_TOKEN}<{x1}><{y1}><{x2}><{y2}>{BOX_END_TOKEN}"


def build_prompt(
    expression: str,
    num_image_tokens: int,
    template: str,
    *,
    with_span: bool = False,
):
    """Reproduces the base processor's ``py_apply_chat_template`` exactly.

    With ``with_span`` also returns the character span of the referring
    expression itself, which the layer router needs: routing the whole text
    region routes mostly the template rather than the expression.

    The span is derived from the same pieces the prompt is assembled from rather
    than searched for afterwards, because searching is wrong in ways that are
    hard to see: short expressions like "a" occur inside "assistant".
    """
    image_block = (
        f"<image 1>{IMAGE_START_TOKEN}"
        f"{IMAGE_CONTEXT_TOKEN * num_image_tokens}"
        f"{IMAGE_END_TOKEN}"
    )
    expression_text = expression.strip().rstrip(".")
    before, _, after = template.partition("{expr}")
    head = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{image_block}{before}"
    )
    prompt = f"{head}{expression_text}{after}<|im_end|>\n<|im_start|>assistant\n"
    if not with_span:
        return prompt
    return prompt, (len(head), len(head) + len(expression_text))


def token_span_for_chars(
    offsets: Sequence[tuple[int, int]], span: tuple[int, int]
) -> tuple[int, int]:
    """Token span covering a character span, as ``[start, end)``.

    Any token that overlaps the character range counts, so a token straddling
    the boundary is kept rather than dropped - losing the first token of a short
    expression matters more than including one character of the template.
    """
    start_char, end_char = span
    indices = [
        index
        for index, (lo, hi) in enumerate(offsets)
        if hi > start_char and lo < end_char
    ]
    if not indices:
        return 0, 0
    return indices[0], indices[-1] + 1


def build_answer(expression: str, boxes: Sequence[Sequence[float]]) -> str:
    """Ground-truth assistant turn for a referring expression.

    An empty box list becomes ``<box>none</box>``, which is how the base VLM
    expresses "no such object" and what gRefCOCO's no-target expressions need.
    """
    ref = f"{REF_START_TOKEN}{expression.strip().rstrip('.')}{REF_END_TOKEN}"
    if len(boxes) == 0:
        return f"{ref}{BOX_START_TOKEN}{NONE_TOKEN}{BOX_END_TOKEN}"
    return ref + "".join(format_box(box) for box in boxes)


def parse_boxes(text: str) -> np.ndarray:
    """Extract normalised ``(x1, y1, x2, y2)`` boxes from a generated string."""
    boxes = [
        [int(g) / COORD_BINS for g in match.groups()]
        for match in BOX_PATTERN.finditer(text)
    ]
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    array = np.asarray(boxes, dtype=np.float32)
    # Guard against inverted coordinates coming out of sampling.
    array[:, [0, 2]] = np.sort(array[:, [0, 2]], axis=1)
    array[:, [1, 3]] = np.sort(array[:, [1, 3]], axis=1)
    return array


def find_box_token_spans(token_ids: Sequence[int], special: SpecialTokenIds) -> np.ndarray:
    """Locate every ``<box> x1 y1 x2 y2 </box>`` run in a token sequence.

    Returns an ``[N, TOKENS_PER_BOX]`` array of absolute positions. Empty boxes
    (``<box>none</box>``) are skipped: they have no contour to evolve.
    """
    ids = np.asarray(token_ids)
    starts = np.flatnonzero(ids == special.box_start)
    spans = []
    for start in starts:
        end = start + TOKENS_PER_BOX - 1
        if end >= len(ids) or ids[end] != special.box_end:
            continue
        coords = ids[start + 1 : end]
        if not np.all((coords >= special.coord_start) & (coords <= special.coord_end)):
            continue
        spans.append(np.arange(start, end + 1))
    if not spans:
        return np.zeros((0, TOKENS_PER_BOX), dtype=np.int64)
    return np.stack(spans).astype(np.int64)


def decode_box_spans(token_ids: Sequence[int], spans: np.ndarray, special: SpecialTokenIds) -> np.ndarray:
    """Normalised boxes for the spans returned by :func:`find_box_token_spans`."""
    if spans.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    ids = np.asarray(token_ids)
    coords = ids[spans[:, 1:5]] - special.coord_start
    boxes = np.clip(coords.astype(np.float32) / COORD_BINS, 0.0, 1.0)
    boxes[:, [0, 2]] = np.sort(boxes[:, [0, 2]], axis=1)
    boxes[:, [1, 3]] = np.sort(boxes[:, [1, 3]], axis=1)
    return boxes
