"""High-resolution feature map that the contour points move on.

The language model sees the image at a stride of 28 px (14 px patches merged
2x2), which carries the semantics but not the boundary. A contour point that
has to land on an object edge needs evidence at a much finer scale, so a small
convolutional stem reads the pixels directly and is fused with the
instruction-conditioned LLM features. The stem is the only full-resolution part
of the network, and it is a few hundred thousand parameters.

Images in a batch have different sizes because the V-L module preserves aspect
ratio. Everything here is stored right/bottom padded to a common size, with the
valid extent tracked separately, so all sampling is done in normalised
coordinates and never reads the padding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import PixelDecoderConfig


class ChannelNorm(nn.Module):
    """LayerNorm across channels at each position, for padding invariance.

    ``GroupNorm`` pools its statistics over the whole spatial map, and the maps
    here are right/bottom padded to whatever the largest image in the batch
    needs. So an image's features depend on its batch-mates, and - because
    inference runs one image at a time with no padding at all - on whether it is
    training or not.

    Normalising over channels at a position touches no other position, so the
    only remaining dependence on padding is the convolutions' own edge effect,
    which is local and bounded by the receptive field.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        normed = (x - mean) * torch.rsqrt(variance + self.eps)
        return normed * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def _norm(dim: int, kind: str) -> nn.Module:
    if kind == "channel":
        return ChannelNorm(dim)
    if kind == "group":
        return nn.GroupNorm(min(32, dim), dim)
    raise ValueError(f"unknown pixel decoder norm {kind!r}")


class ConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1, norm: str = "group") -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, stride=stride, padding=1, bias=False)
        self.norm = _norm(out_dim, norm)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, norm: str = "group") -> None:
        super().__init__()
        self.block = nn.Sequential(ConvBlock(dim, dim, norm=norm), ConvBlock(dim, dim, norm=norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Pyramid(nn.Module):
    """Downsample, think with a wide view, come back up.

    The stem's convolutional support is short relative to how far a point may
    need to travel, and the language grid has range but a coarse stride. Each
    level here doubles the stride, which multiplies the receptive field while the
    cost falls, and the skips carry the fine detail back.
    """

    def __init__(self, dim: int, levels: int, depth: int, norm: str = "group") -> None:
        super().__init__()
        self.down = nn.ModuleList(
            nn.Sequential(
                ConvBlock(dim, dim, stride=2, norm=norm),
                *[ConvBlock(dim, dim, norm=norm) for _ in range(depth - 1)],
            )
            for _ in range(levels)
        )
        # Takes the upsampled coarse features concatenated with the skip.
        self.up = nn.ModuleList(
            nn.Sequential(
                ConvBlock(2 * dim, dim, norm=norm),
                *[ConvBlock(dim, dim, norm=norm) for _ in range(depth - 1)],
            )
            for _ in range(levels)
        )

    def forward(self, features: torch.Tensor, fuse=None) -> torch.Tensor:
        skips = []
        for block in self.down:
            skips.append(features)
            features = block(features)

        # The language grid is fused at the bottom rather than at full
        # resolution. It arrives at stride 28, so upsampling it 14x to meet the
        # stem was mostly interpolation; here the two are within a factor of two
        # of each other and the decoder carries the result back up itself.
        if fuse is not None:
            features = fuse(features)

        for block, skip in zip(reversed(self.up), reversed(skips)):
            features = F.interpolate(
                features, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            features = block(torch.cat([features, skip], dim=1))
        return features


class PixelDecoder(nn.Module):
    def __init__(self, config: PixelDecoderConfig, llm_dim: int) -> None:
        super().__init__()
        self.config = config
        self.stride = config.output_stride
        dim = config.hidden_dim

        layers: list[nn.Module] = []
        in_dim = 3
        remaining = self.stride
        for out_dim in config.stem_dims:
            stride = 2 if remaining > 1 else 1
            layers.append(ConvBlock(in_dim, out_dim, stride=stride, norm=config.norm))
            remaining //= stride
            in_dim = out_dim
        layers.append(nn.Conv2d(in_dim, dim, 1))
        self.stem = nn.Sequential(*layers)

        self.llm_projection = (
            nn.Sequential(_norm(llm_dim, config.norm), nn.Conv2d(llm_dim, dim, 1))
            if config.use_llm_features
            else None
        )
        self.pyramid = (
            Pyramid(dim, config.pyramid_levels, config.pyramid_depth, config.norm)
            if config.pyramid_levels > 0
            else None
        )
        self.fuse = nn.Sequential(
            *[ResidualBlock(dim, config.norm) for _ in range(config.num_fusion_blocks)]
        )
        self.output_norm = _norm(dim, config.norm)
        self.dropout = nn.Dropout2d(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        images: torch.Tensor,
        image_sizes: torch.Tensor,
        visual_grid: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: ``[B, 3, H, W]`` padded, normalised to the backbone's range.
            image_sizes: ``[B, 2]`` valid ``(height, width)`` per image.
            visual_grid: per-image ``[D_llm, h, w]`` language-model features.

        Returns:
            ``([B, C, H/s, W/s], [B, 2])`` feature map and valid feature sizes.
        """
        images = images.to(dtype=self.stem[0].conv.weight.dtype)
        if self.config.max_stem_side > 0:
            longest = int(max(images.shape[-2], images.shape[-1]))
            if longest > self.config.max_stem_side:
                scale = self.config.max_stem_side / longest
                target = (
                    _round_to(images.shape[-2] * scale, self.stride),
                    _round_to(images.shape[-1] * scale, self.stride),
                )
                image_sizes = torch.stack(
                    [
                        (image_sizes[:, 0].float() * target[0] / images.shape[-2]).round().long(),
                        (image_sizes[:, 1].float() * target[1] / images.shape[-1]).round().long(),
                    ],
                    dim=1,
                )
                images = F.interpolate(images, size=target, mode="bilinear", align_corners=False)

        features = self.stem(images)
        feature_sizes = torch.stack(
            [
                (image_sizes[:, 0] // self.stride).clamp(min=1),
                (image_sizes[:, 1] // self.stride).clamp(min=1),
            ],
            dim=1,
        )

        def fuse_language(coarse: torch.Tensor) -> torch.Tensor:
            if self.llm_projection is None or visual_grid is None:
                return coarse
            # Follow the same halving the convolutions do - a stride-2 conv with
            # padding 1 takes n to ceil(n/2) - rather than dividing by the ratio
            # of the padded widths. Those disagree whenever a size is odd, and the
            # valid extent then comes out one row larger than the map it indexes.
            sizes = feature_sizes
            for _ in range(len(self.pyramid.down)):
                sizes = (sizes + 1) // 2
            limit = torch.tensor(coarse.shape[-2:], device=sizes.device)
            sizes = sizes.clamp(min=1).minimum(limit[None])
            return coarse + self._upsample_llm_features(visual_grid, coarse, sizes)

        if self.pyramid is not None:
            features = self.pyramid(features, fuse=fuse_language)
        elif self.llm_projection is not None and visual_grid is not None:
            features = features + self._upsample_llm_features(
                visual_grid, features, feature_sizes
            )

        features = self.output_norm(self.fuse(features))
        return self.dropout(features), feature_sizes

    def _upsample_llm_features(
        self,
        visual_grid: list[torch.Tensor],
        reference: torch.Tensor,
        feature_sizes: torch.Tensor,
    ) -> torch.Tensor:
        buffer = torch.zeros_like(reference)
        for index, grid in enumerate(visual_grid):
            projected = self.llm_projection(grid[None].to(reference.dtype))
            height, width = int(feature_sizes[index, 0]), int(feature_sizes[index, 1])
            resized = F.interpolate(
                projected, size=(height, width), mode="bilinear", align_corners=False
            )
            buffer[index, :, :height, :width] = resized[0]
        return buffer


def _round_to(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #


class DistanceHead(nn.Module):
    """How far is the boundary, read from one feature vector.

    Deliberately small and deliberately shallow. It exists to put a loss one
    layer away from the features so the pixel decoder is asked the question
    directly, not to be good at answering it on its own - a head with capacity to
    spare would learn the distance itself and let the features stay as they are,
    which is the opposite of the point.

    Output is a distance in normalised image coordinates, not in units of the
    box. The feature map is one per image and knows nothing about instances, so
    the same location can fall inside two instances' regions; asked for a
    box-relative distance it would owe two different answers there. Dividing by
    the box happens in the loss, where the instance is known.
    """

    def __init__(self, feature_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``[..., C] -> [...]``, signed, positive inside."""
        return self.net(features).squeeze(-1)


def sample_points(
    features: torch.Tensor,
    points: torch.Tensor,
    batch_index: torch.Tensor,
    feature_sizes: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly read ``features`` at normalised contour points.

    Args:
        features: ``[B, C, H, W]`` padded feature map.
        points: ``[M, P, 2]`` normalised ``(x, y)`` in ``[0, 1]``.
        batch_index: ``[M]`` image each instance belongs to.
        feature_sizes: ``[B, 2]`` valid ``(height, width)`` of the feature map.

    Returns:
        ``[M, P, C]``.
    """
    if points.shape[0] == 0:
        return points.new_zeros((0, points.shape[1], features.shape[1]))

    padded_h, padded_w = features.shape[-2], features.shape[-1]
    valid = feature_sizes[batch_index].to(points.dtype)  # [M, 2] as (h, w)
    scale = torch.stack([valid[:, 1] / padded_w, valid[:, 0] / padded_h], dim=1)
    grid = (points.clamp(0.0, 1.0) * scale[:, None, :]) * 2.0 - 1.0

    # One grid_sample call per image keeps the batch dimension aligned.
    output = points.new_zeros((points.shape[0], points.shape[1], features.shape[1]))
    for image in torch.unique(batch_index):
        selection = batch_index == image
        sampled = F.grid_sample(
            features[image : image + 1].to(points.dtype),
            grid[selection].unsqueeze(0).reshape(1, -1, points.shape[1], 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        output[selection] = sampled[0].permute(1, 2, 0)
    return output


def roi_features(
    features: torch.Tensor,
    boxes: torch.Tensor,
    batch_index: torch.Tensor,
    feature_sizes: torch.Tensor,
    output_size: int,
    expand: float = 1.0,
) -> torch.Tensor:
    """ROIAlign the feature map inside each (expanded) box.

    Gives the decoder a fixed-size view of the object it is tracing, which is
    what makes a variable number of instances per image a non-issue: every
    instance becomes an independent row with identical shapes.

    Returns ``[M, C, K, K]``.
    """
    from torchvision.ops import roi_align

    if boxes.shape[0] == 0:
        return features.new_zeros((0, features.shape[1], output_size, output_size))

    center = torch.stack(
        [(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], dim=1
    )
    half = torch.stack(
        [(boxes[:, 2] - boxes[:, 0]) * 0.5, (boxes[:, 3] - boxes[:, 1]) * 0.5], dim=1
    ) * expand
    expanded = torch.cat([center - half, center + half], dim=1).clamp(0.0, 1.0)

    valid = feature_sizes[batch_index].to(boxes.dtype)  # (h, w)
    scale = torch.stack([valid[:, 1], valid[:, 0], valid[:, 1], valid[:, 0]], dim=1)
    absolute = expanded * scale
    rois = torch.cat([batch_index[:, None].to(boxes.dtype), absolute], dim=1)

    return roi_align(
        features.to(boxes.dtype),
        rois,
        output_size=(output_size, output_size),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )
