#!/usr/bin/env python3
"""Joint grounding/continuous-action RL, or explicit legacy grounding-only RL.

Ranks sample independent rollout groups. Gradients are averaged manually before
clipping; no DDP reducer is mixed with variable-length per-rank backward loops.
The backbone retains its configured LoRA and freezing policy and is trained by
the answer-token surrogate. Joint training additionally optimizes pixel
features, action decoder, and router using sampled action likelihoods; those
reach the backbone only when ``train.segmentation_updates_backbone`` says so.
Inference continues to use deterministic action means.

Learning rates warm up linearly over ``train.warmup_steps`` and then stay
constant; a group whose AdamW moments were inherited from the init checkpoint
(``rl.inherit_optimizer``) skips the warm-up. Each parameter group is clipped
on its own, so a spiky action gradient cannot shrink the grounding update.

best.pt is selected by the configured evaluation metric, against a bar set by
evaluating the starting weights; last.pt is saved on evaluation/save intervals;
final.pt is the last update. All contain the full model, optimizer groups and
config. Use init_from, not resume, when changing from a legacy optimizer to
joint training. RNG streams are not checkpointed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contourvla.config import ContourVLAConfig
from contourvla.data.datasets import load_referring_samples
from contourvla.engine import evaluate, spread_samples
from contourvla.models.contourvla import ContourVLA
from contourvla.rl.loop import rl_step
from contourvla.rl.rollout import ReferenceSnapshot

REFERENCES = ("lora_off", "initial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="+", action="extend", default=[], dest="overrides")
    return parser.parse_args()


def apply_overrides(config: ContourVLAConfig, overrides: list[str]) -> None:
    for override in overrides:
        key, _, raw = override.partition("=")
        section, _, name = key.partition(".")
        target = getattr(config, section)
        current = getattr(target, name)
        if isinstance(current, bool):
            value = raw.lower() in ("1", "true", "yes")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(float(raw))
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw
        setattr(target, name, value)


def configure_rl_parameters(model, config):
    """Keep backbone's configured LoRA/freezes; train all active geometric paths."""
    settings = config.rl
    if settings.group < 2 or config.train.grad_accum < 1:
        raise ValueError("RL requires group >= 2 and grad_accum >= 1")
    for name in ("advantage_floor", "action_advantage_floor"):
        value = getattr(settings, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if settings.reference not in REFERENCES:
        raise ValueError(f"rl.reference must be one of {REFERENCES}")
    if settings.joint_training:
        values = (settings.action_std, settings.action_loss_weight, config.train.lr)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("joint RL requires positive action_std, action_loss_weight and train.lr")
    model.contour_modules.requires_grad_(settings.joint_training)
    # An auxiliary-only SDF head is not part of the action likelihood. It stays
    # fixed unless its existing feedback channel participates in the policy.
    if model.distance_head is not None:
        feedback = settings.joint_training and config.action_decoder.sdf_feedback
        model.distance_head.requires_grad_(feedback)
        if feedback:
            model.action_decoder.config.sdf_detach = False
    groups = model.parameter_groups(config.train.lr, config.train.backbone_lr,
                                    config.train.weight_decay)
    backbone = {id(p) for p in model.backbone.parameters()}
    for group in groups:
        group["name"] = "backbone" if id(group["params"][0]) in backbone else "contour"
        group["base_lr"] = group["lr"]
        group["warmup"] = True
    return groups


def set_learning_rates(optimizer, step: int, warmup_steps: int) -> None:
    """Linear warm-up to each group's base rate, then constant."""
    for group in optimizer.param_groups:
        base = group.get("base_lr", group["lr"])
        ramp = 1.0
        if warmup_steps > 0 and group.get("warmup", True):
            ramp = min(1.0, step / warmup_steps)
        group["lr"] = base * ramp


def clip_groups(optimizer, max_norm: float) -> dict[str, float]:
    """Clip each parameter group separately; returns the pre-clip norms."""
    norms = {}
    for index, group in enumerate(optimizer.param_groups):
        parameters = [p for p in group["params"] if p.grad is not None]
        if not parameters:
            continue
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
        norms[f"grad_{group.get('name', index)}"] = float(norm)
    return norms


def average_gradients(parameters):
    """Average before clipping; globally unused parameters retain grad=None."""
    if not dist.is_initialized():
        return
    parameters = list(parameters)
    if not parameters:
        return
    active = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32,
                          device=parameters[0].device)
    dist.all_reduce(active, op=dist.ReduceOp.SUM)
    for parameter, used in zip(parameters, active.tolist()):
        if not used:
            parameter.grad = None
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(dist.get_world_size())


def rl_parameter_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def save_rl_checkpoint(path, model, optimizer, config, step, history, best_value):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "parameter_names": rl_parameter_names(model, optimizer),
                "config": config.to_dict(), "step": step, "history": history,
                "router_step": getattr(model, "_router_step", 0),
                "best_metric": config.train.best_metric, "best_value": best_value}, path)


def restore_rl_checkpoint(path, model, optimizer, config):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "parameter_names" in payload and payload["parameter_names"] != rl_parameter_names(model, optimizer):
        raise ValueError("RL optimizer parameter scope changed; use train.init_from for weight-only initialization")
    saved_rl = payload.get("config", {}).get("rl", {})
    for key in ("joint_training", "action_std"):
        if key in saved_rl and saved_rl[key] != getattr(config.rl, key):
            raise ValueError(f"RL {key} changed; use train.init_from instead of resume")
    if "optimizer" not in payload:
        raise ValueError("RL resume needs optimizer state; use train.init_from for this checkpoint")
    try:
        optimizer.load_state_dict(payload["optimizer"])
    except ValueError as error:
        raise ValueError("incompatible RL optimizer; use train.init_from for the old checkpoint") from error
    model.load_state_dict(payload["model"], strict=True)
    model._router_step = int(payload.get("router_step", 0))
    return payload


def inherit_optimizer_state(optimizer, model, payload, log=print) -> dict[str, int]:
    """Copy AdamW moments from an init checkpoint into matching parameters.

    Matched by parameter name. A checkpoint written before names were stored
    is read as the grounding-only stage wrote it: one group over the trainable
    backbone tensors in module order, accepted only if the count and every
    shape agree. A group whose tensors were all matched skips the warm-up.
    """
    saved = payload.get("optimizer")
    if not saved or not saved.get("state"):
        log("[inherit] init checkpoint carries no optimizer state; AdamW starts at zero")
        return {}
    saved_names = payload.get("parameter_names")
    if saved_names is None:
        backbone = [f"backbone.{n}" for n, p in model.backbone.named_parameters() if p.requires_grad]
        groups = saved["param_groups"]
        if len(groups) != 1 or len(groups[0]["params"]) != len(backbone):
            log(f"[inherit] unnamed optimizer with {[len(g['params']) for g in groups]} tensors "
                f"does not match the {len(backbone)} trainable backbone tensors; not inherited")
            return {}
        saved_names = [backbone]
    by_name = {}
    for names, group in zip(saved_names, saved["param_groups"]):
        for name, index in zip(names, group["params"]):
            if index in saved["state"]:
                by_name[name] = saved["state"][index]
    names = {id(p): n for n, p in model.named_parameters()}
    copied = {}
    for group in optimizer.param_groups:
        count = 0
        for parameter in group["params"]:
            entry = by_name.get(names[id(parameter)])
            if entry is None or entry["exp_avg"].shape != parameter.shape:
                continue
            optimizer.state[parameter] = {
                "step": torch.tensor(float(entry["step"]), dtype=torch.float32),
                "exp_avg": entry["exp_avg"].to(parameter.device, parameter.dtype).clone(),
                "exp_avg_sq": entry["exp_avg_sq"].to(parameter.device, parameter.dtype).clone(),
            }
            count += 1
        if count and count == len(group["params"]):
            group["warmup"] = False
        copied[group.get("name", str(len(copied)))] = count
        log(f"[inherit] {group.get('name')}: {count}/{len(group['params'])} AdamW states "
            f"from the init checkpoint; warm-up {'on' if group.get('warmup', True) else 'off'}")
    return copied


def reference_snapshot(model, config, log=print) -> ReferenceSnapshot | None:
    """The KL anchor for ``rl.reference == "initial"``: the stage's starting weights."""
    if config.rl.reference != "initial":
        return None
    named = [(n, p) for n, p in model.backbone.named_parameters() if p.requires_grad]
    values = None
    if config.train.resume:
        if not config.train.init_from:
            raise ValueError("an initial-policy reference on resume needs train.init_from")
        state = torch.load(config.train.init_from, map_location="cpu", weights_only=False)["model"]
        values = [state[f"backbone.{n}"] for n, _ in named]
        del state
    snapshot = ReferenceSnapshot([p for _, p in named], values)
    log(f"[reference] initial policy: {len(named)} backbone tensors, "
        f"{sum(p.numel() for _, p in named) / 1e6:.1f}M values")
    return snapshot


def existing_samples(samples):
    """Drop samples whose image is missing, checking each path once."""
    paths = {s.image_path for s in samples}
    present = {path for path in paths if os.path.exists(path)}
    return [s for s in samples if s.image_path in present]


def main(load_train=None, load_eval=None) -> None:
    """``load_train`` / ``load_eval`` (``(config, verbose) -> samples``) replace the
    default split loaders, so a wrapper can compose its own data mixture."""
    args = parse_args()
    config = ContourVLAConfig.load(args.config)
    apply_overrides(config, args.overrides)

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local)
    if world > 1:
        # Longer than the NCCL default. A step is many autoregressive decodes,
        # and periodic evaluation generates as well, so ranks can reach the
        # gradient all-reduce minutes apart.
        dist.init_process_group(
            "nccl", device_id=device, timeout=timedelta(minutes=60)
        )
        torch.cuda.set_device(local)

    def log(message: str) -> None:
        if rank == 0:
            print(message, flush=True)

    torch.manual_seed(config.train.seed)
    model = ContourVLA(config).to(device)
    if config.train.init_from and not config.train.resume:
        from contourvla.engine import Trainer

        seed = Trainer.__new__(Trainer)
        seed.core = model
        seed.step = 0
        seed.initialize_from(config.train.init_from, config.train.init_keep,
                             config.train.init_drop, log_fn=log)

    groups = configure_rl_parameters(model, config)
    trainable = [p for group in groups for p in group["params"]]
    log(f"joint_training={config.rl.joint_training}: training {sum(p.numel() for p in trainable) / 1e6:.1f}M parameters")
    for group in groups:
        log(f"  group {group['name']}: lr={group['lr']}, tensors={len(group['params'])}")
    optimizer = torch.optim.AdamW(groups)
    core = model
    if world > 1:
        with torch.no_grad():
            for tensor in list(core.parameters()) + list(core.buffers()):
                dist.broadcast(tensor, src=0)
    if config.rl.inherit_optimizer and config.train.init_from and not config.train.resume:
        payload = torch.load(config.train.init_from, map_location="cpu", weights_only=False)
        inherit_optimizer_state(optimizer, core, payload, log)
        del payload
    torch.manual_seed(config.train.seed + rank)

    if load_train is None:
        samples = load_referring_samples(config.data, verbose=rank == 0)
    else:
        samples = load_train(config, rank == 0)
    train_samples = existing_samples(samples)
    # Split once so no-target images can be drawn at a chosen rate instead of
    # their natural 5.5%. `pick_sample` below reads these.
    positives = [s for s in train_samples if len(s.boxes) > 0]
    negatives = [s for s in train_samples if len(s.boxes) == 0]
    share = config.rl.negative_share
    period = int(round(1.0 / share)) if share > 0 and negatives else 0

    def pick_sample(index: int):
        """Sample for a global step index, oversampling refusals to `share`.

        Indexed rather than shuffled so every rank derives the same stream from
        the same arithmetic, and so a resumed run continues it rather than
        starting the sequence again.
        """
        if not period:
            return train_samples[index % len(train_samples)]
        if index % period == 0:
            return negatives[(index // period) % len(negatives)]
        drawn = index - (index // period) - 1
        return positives[drawn % len(positives)]

    if load_eval is None:
        val_config = copy.deepcopy(config.data)
        val_config.split = config.data.val_split
        val_samples = load_referring_samples(val_config, verbose=False)
    else:
        val_samples = load_eval(config, False)
    log(f"{len(train_samples)} training samples, {len(val_samples)} for evaluation")
    if period:
        log(f"no-target oversampled to 1 in {period} steps "
            f"({len(negatives)} available, {len(negatives) / len(train_samples):.1%} naturally)")

    os.makedirs(config.train.output_dir, exist_ok=True)
    if rank == 0:
        # train.py writes this and evaluate.py checks a checkpoint against it to
        # refuse a mismatched architecture. Without it an RL checkpoint can only
        # be evaluated by naming the yaml by hand and hoping it is the right one.
        config.save(os.path.join(config.train.output_dir, "config.json"))
    history: list[dict] = []
    # Best held-out score so far, for best.pt below. None means "nothing measured
    # yet", which is distinct from a genuine 0.0.
    best_value: float | None = None

    def run_eval():
        # evaluate() seeds the global RNG for reproducible decoding; forked so
        # it does not also reset the rollout stream to the same state each time.
        with torch.random.fork_rng(devices=[local] if torch.cuda.is_available() else []):
            return evaluate(
                core, val_samples, max_samples=config.train.eval_samples,
                log_fn=log, rank=rank, world_size=world,
            )

    # A step here costs a minute and the stage runs for a day, so an
    # interruption that cannot be resumed is expensive. last.pt now carries the
    # optimizer and the step, and this picks up from the step after it.
    first_step = 1
    if config.train.resume:
        payload = restore_rl_checkpoint(config.train.resume, core, optimizer, config)
        history = payload.get("history", [])
        if payload.get("best_metric") == config.train.best_metric:
            best_value = payload.get("best_value")
        first_step = int(payload.get("step", 0)) + 1
        del payload
        log(f"resumed from {config.train.resume} at step {first_step - 1}")
        # Recover the bar an existing best.pt already set, or a resumed run would
        # overwrite a better checkpoint with the first eval it happens to run.
        best_path = os.path.join(config.train.output_dir, "best.pt")
        if os.path.exists(best_path):
            kept = torch.load(best_path, map_location="cpu", weights_only=False)
            if kept.get("best_metric") == config.train.best_metric:
                best_value = kept.get("best_value")
                if best_value is not None:
                    log(f"existing best.pt holds {config.train.best_metric}="
                        f"{best_value:.4f} at step {kept.get('step')}")
            del kept
    core.rl_reference = reference_snapshot(core, config, log)
    if not config.train.resume and config.train.eval_every and val_samples:
        # The starting weights set the bar, so best.pt is only written by a step
        # that actually beats them.
        metrics = run_eval()
        metric = config.train.best_metric
        if metrics.get(metric) is not None:
            best_value = float(metrics[metric])
            log(f"[start] {metric}={best_value:.4f}; best.pt is written only above this")
        history.append({"step": 0, **{f"eval_{k}": v for k, v in metrics.items()}})

    start = time.time()
    for step in range(first_step, config.train.max_steps + 1):
        set_learning_rates(optimizer, step, config.train.warmup_steps)
        optimizer.zero_grad(set_to_none=True)
        tally: dict[str, float] = {}
        for micro in range(config.train.grad_accum):
            index = ((step - 1) * config.train.grad_accum + micro) * world + rank
            sample = pick_sample(index)
            image = Image.open(sample.image_path).convert("RGB")
            stats = rl_step(core, image, sample, config.rl)
            for key, value in stats.items():
                tally[key] = tally.get(key, 0.0) + value / config.train.grad_accum

        for parameter in trainable:
            if parameter.grad is not None:
                parameter.grad.div_(config.train.grad_accum)
        average_gradients(trainable)
        tally.update(clip_groups(optimizer, config.train.grad_clip))
        optimizer.step()

        if step % config.train.log_every == 0:
            # Averaged over ranks before printing. Without this the line is one
            # rank's two rollout groups out of the thirty-two that actually
            # produced the step's gradient, and the metric that matters most -
            # group_used, the share of groups with enough spread to learn from -
            # reads 0 or 1 at random. The gradient was always all-reduced; only
            # the reporting was a single sample.
            if world > 1 and tally:
                names = sorted(tally)
                packed = torch.tensor(
                    [tally[k] for k in names], dtype=torch.float32, device=device
                )
                dist.all_reduce(packed, op=dist.ReduceOp.AVG)
                tally = dict(zip(names, packed.tolist()))
            elapsed = (time.time() - start) / max(step - first_step + 1, 1)
            learning_rates = " ".join(
                f"lr_{g.get('name', i)}={g['lr']:.2e}" for i, g in enumerate(optimizer.param_groups)
            )
            log(f"step {step}/{config.train.max_steps} "
                + " ".join(f"{k}={v:.4f}" for k, v in sorted(tally.items()))
                + f" {learning_rates} ({elapsed:.1f}s/step)")
            history.append({"step": step, **tally})

        if config.train.eval_every and step % config.train.eval_every == 0:
            metrics = run_eval()
            history.append({"step": step, **{f"eval_{k}": v for k, v in metrics.items()}})
            metric = config.train.best_metric
            value = metrics.get(metric) if metric else None
            if value is not None and (best_value is None or value > best_value):
                best_value = float(value)
                if rank == 0:
                    save_rl_checkpoint(os.path.join(config.train.output_dir, "best.pt"),
                                       core, optimizer, config, step, history, best_value)
                log(f"[best] {metric}={best_value:.4f} at step {step} -> best.pt")

        should_save = (
            (config.train.save_every and step % config.train.save_every == 0)
            or (config.train.eval_every and step % config.train.eval_every == 0)
        )
        if rank == 0 and should_save:
            save_rl_checkpoint(os.path.join(config.train.output_dir, "last.pt"),
                               core, optimizer, config, step, history, best_value)
            with open(os.path.join(config.train.output_dir, "history.json"), "w") as handle:
                json.dump(history, handle, indent=2)

    if rank == 0:
        save_rl_checkpoint(os.path.join(config.train.output_dir, "final.pt"),
                           core, optimizer, config, config.train.max_steps, history, best_value)
        with open(os.path.join(config.train.output_dir, "history.json"), "w") as handle:
            json.dump(history, handle, indent=2)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
