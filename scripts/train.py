#!/usr/bin/env python3
"""Train ContourVLA.

Single GPU::

    python scripts/train.py --config configs/sft.yaml

Multi GPU::

    torchrun --nproc_per_node=8 scripts/train.py --config configs/sft.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contourvla.config import ContourVLAConfig
from contourvla.data.datasets import (
    ContourReferringDataset,
    SyntheticContourDataset,
    load_jsonl_samples,
    load_referring_samples,
)
from contourvla.engine import Trainer, WandbLogger, build_dataloader
from contourvla.models.contourvla import ContourVLA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    # `extend` rather than the default `store`: the launch script passes one
    # `--set` and the caller may pass another, and the second must not silently
    # replace the first.
    parser.add_argument(
        "--set",
        nargs="+",
        action="extend",
        default=[],
        dest="overrides",
        help="section.key=value overrides, repeatable",
    )
    return parser.parse_args()


def apply_overrides(config: ContourVLAConfig, overrides: list[str]) -> None:
    for override in overrides:
        path, separator, raw = override.partition("=")
        section, _, key = path.partition(".")
        if not separator or not key or not hasattr(config, section):
            raise ValueError(f"malformed override {override!r}; expected section.key=value")
        target = getattr(config, section)
        if not hasattr(target, key):
            raise ValueError(f"unknown config key {section}.{key}")
        current = getattr(target, key)
        if isinstance(current, bool):
            value = raw.lower() in ("1", "true", "yes")
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        setattr(target, key, value)


@contextlib.contextmanager
def main_process_first(rank: int, world_size: int):
    """Let rank 0 run the block alone, then release the others.

    Used for anything that writes to a shared cache: whichever rank gets there
    first pays the cost and the rest read the finished result.
    """
    if world_size <= 1:
        yield
        return
    if rank != 0:
        dist.barrier()
    try:
        yield
    finally:
        if rank == 0:
            dist.barrier()


def install_stack_dumper() -> None:
    """``kill -USR1 <pid>`` makes a rank print where it is.

    A distributed job that stops making progress gives you nothing to go on: no
    traceback, no exit code, just silence until the collective times out. This
    costs nothing to install and turns "it hung" into a stack trace from every
    thread of whichever rank you ask.
    """
    import faulthandler
    import signal

    faulthandler.enable()
    if hasattr(faulthandler, "register"):
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


def setup_distributed() -> tuple[int, int, torch.device]:
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Binding the device up front lets `barrier()` pick the right one instead of
    # guessing, which NCCL warns can hang.
    #
    # The default ten minutes is not enough headroom here. Two phases run on one
    # rank while the others wait inside a collective: the first parse of the
    # annotations, and the periodic visualisation. Neither is close to an hour,
    # but a cold shared filesystem can turn a one-minute read into a long one,
    # and the failure mode is the whole job dying rather than a slow step.
    dist.init_process_group(
        backend="nccl", device_id=device, timeout=timedelta(minutes=60)
    )
    return rank, dist.get_world_size(), device


def cache_path(config: ContourVLAConfig, split: str) -> str | None:
    if not config.data.cache_dir:
        return None
    return os.path.join(
        config.data.cache_dir,
        f"{config.data.dataset}_{config.data.split_by}_{split}.pkl",
    )


def load_split(config: ContourVLAConfig, split: str, verbose: bool):
    data_config = copy.deepcopy(config.data)
    data_config.split = split
    if data_config.dataset == "jsonl":
        return load_jsonl_samples(data_config.ref_file, data_config.image_root)
    return load_referring_samples(
        data_config, cache_path=cache_path(config, split), verbose=verbose
    )


def main() -> None:
    args = parse_args()
    config = ContourVLAConfig.load(args.config) if args.config else ContourVLAConfig()
    apply_overrides(config, args.overrides)

    install_stack_dumper()
    rank, world_size, device = setup_distributed()
    torch.manual_seed(config.train.seed + rank)

    def log(message: str) -> None:
        if rank == 0:
            print(message, flush=True)

    if rank == 0:
        os.makedirs(config.train.output_dir, exist_ok=True)
        config.save(os.path.join(config.train.output_dir, "config.json"))

    log(f"building ContourVLA from {config.backbone.checkpoint_path}")
    model = ContourVLA(config).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"parameters: {total / 1e9:.2f}B total, {trainable / 1e6:.1f}M trainable")

    backbone = model.backbone
    shared = dict(
        processor=backbone.processor,
        tokenizer=backbone.tokenizer,
        special=backbone.special,
        data_config=config.data,
        contour_config=config.contour,
        merge_size=backbone.merge_size,
    )

    val_samples = []
    if config.data.dataset == "synthetic":
        dataset = SyntheticContourDataset(length=256, **shared)
    else:
        # Rank 0 parses the JSON and writes the cache; the others wait and read it.
        with main_process_first(rank, world_size):
            train_samples = load_split(config, config.data.split, verbose=rank == 0)
            needs_val = config.train.eval_every or config.visualization.source in ("val", "both")
            if needs_val and config.data.val_split:
                val_samples = load_split(config, config.data.val_split, verbose=rank == 0)
        dataset = ContourReferringDataset(samples=train_samples, training=True, **shared)

    log(f"dataset: {len(dataset)} train samples, {len(val_samples)} val samples")

    loader = build_dataloader(
        dataset,
        config,
        pad_token_id=backbone.special.pad,
        shuffle=True,
        distributed=world_size > 1,
    )

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    logger = WandbLogger(config) if rank == 0 else None

    trainer = Trainer(
        model,
        config,
        loader,
        device,
        rank=rank,
        world_size=world_size,
        val_samples=val_samples,
        logger=logger,
    )
    if config.train.resume and config.train.init_from:
        raise ValueError(
            "set either train.resume (continue an interrupted run) or "
            "train.init_from (start a new run from old weights), not both"
        )
    if config.train.resume:
        trainer.load(config.train.resume)
        log(f"resumed from {config.train.resume} at step {trainer.step}")
    elif config.train.init_from:
        trainer.initialize_from(
            config.train.init_from,
            config.train.init_keep,
            config.train.init_drop,
            log_fn=log,
        )

    try:
        trainer.train(log_fn=log)
    finally:
        if logger is not None:
            logger.finish()
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
