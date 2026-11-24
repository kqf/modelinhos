import re
import warnings
from pathlib import Path
from typing import Protocol, TypeVar

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

M = TypeVar("M", bound=torch.nn.Module)


class Weights(Protocol):
    def __call__(self, model: M, /) -> M: ...


def from_scratch(model: M) -> M:
    """The identity loader: keep the freshly initialized parameters."""
    return model


def load_with_mismatch(model, pretrained_state_dict):
    def repeat(pretrained_param, model_param):
        if pretrained_param.shape == model_param.shape:
            return pretrained_param

        ns = model_param.shape
        expanded = pretrained_param
        for dim in range(len(ns)):
            # Work on `expanded` throughout: narrowing from the original
            # would discard expansions already applied to earlier dims.
            # Ceil the repeat count so non-divisible targets overshoot;
            # the narrow below trims to the exact size.
            if expanded.shape[dim] < ns[dim]:
                repeats = -(-ns[dim] // expanded.shape[dim])
                expanded = expanded.repeat_interleave(repeats, dim=dim)
            if expanded.shape[dim] > ns[dim]:
                expanded = torch.narrow(expanded, dim, 0, ns[dim])
        return expanded

    model_state_dict = model.state_dict()

    for name, pretrained_param in pretrained_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name] = repeat(
                pretrained_param.clone(),
                model_state_dict[name],
            )

    model.load_state_dict(model_state_dict)
    return model


def modernize_fpn(state: dict) -> dict:
    """Checkpoints predating torchvision's Conv2dNormActivation FPN
    blocks (e.g. the FCOS COCO weights) address the lateral convs as
    fpn.{inner,layer}_blocks.N.weight; current models nest them one
    level deeper (...N.0.weight). torchvision performs this rename in
    FeaturePyramidNetwork._load_from_state_dict, a hook that
    load_with_mismatch's plain dict lookup bypasses -- so redo it here.
    Modern keys don't match the pattern and pass through unchanged."""
    pattern = re.compile(
        r"(.*fpn\.(?:inner|layer)_blocks\.\d+\.)(weight|bias)$"
    )
    return {
        pattern.sub(r"\g<1>0.\g<2>", name): param
        for name, param in state.items()
    }


def state_dict(source, progress: bool = True) -> dict:
    """Normalize a weights source into a plain state dict. A source is
    either a torchvision-style weights object (anything with
    .get_state_dict()) or a checkpoint path. Engine checkpoints hold the
    DetectionModel wrapper's dict (model.-prefixed keys); the prefix is
    stripped -- and legacy FPN keys renamed -- so the result always
    addresses the raw model that build_model produces."""
    if isinstance(source, str | Path):
        state = torch.load(source, map_location="cpu", weights_only=True)
        consume_prefix_in_state_dict_if_present(state, "model.")
        return modernize_fpn(state)
    return modernize_fpn(source.get_state_dict(progress=progress))


class PartialWarmStart(UserWarning):
    """warm_start left parameters outside the expected-fresh prefixes
    at their initialization or reshaped them to fit."""


def signature(state: dict) -> set[tuple[str, tuple[int, ...]]]:
    return {
        (name, tuple(param.shape))
        for name, param in state.items()
        # BatchNorm fills in the counter when a checkpoint lacks it
        if not name.endswith("num_batches_tracked")
    }


def unmatched(
    model_state: dict,
    checkpoint: dict,
    fresh: tuple[str, ...] = (),
) -> set[tuple[str, tuple[int, ...]]]:
    """Model parameters the checkpoint does not cover as-is: either
    absent from it or shaped differently (load_with_mismatch would
    repeat/trim those). Names under the `fresh` prefixes are expected
    to deviate and are skipped."""
    return {
        (name, shape)
        for name, shape in signature(model_state) - signature(checkpoint)
        if not name.startswith(fresh)
    }


def warm_start(
    source,
    fresh: tuple[str, ...] = (),
    progress: bool = True,
) -> Weights:
    """Mismatch-tolerant loader for training: the checkpoint is a
    starting point, not the final model, so parameters sized for a
    different label set are patched by load_with_mismatch (repeat/trim).
    Parameters under the `fresh` prefixes (typically the heads) are
    expected to deviate; any other deviation raises PartialWarmStart.
    Never use it for evaluation or export -- a mismatch there means the
    rebuilt model is not the trained one; that is what restore guards."""

    def load(model: M) -> M:
        checkpoint = state_dict(source, progress)
        if missed := unmatched(model.state_dict(), checkpoint, fresh):
            listed = "\n".join(
                f"  {name} {list(shape)}" for name, shape in sorted(missed)
            )
            warnings.warn(
                f"warm_start: {len(missed)} parameter(s) outside {fresh} "
                f"are missing from the checkpoint or reshaped:\n{listed}",
                PartialWarmStart,
                stacklevel=2,
            )
        return load_with_mismatch(model, checkpoint)

    return load


def restore(source, progress: bool = True) -> Weights:
    """Strict loader for evaluation and export: the checkpoint IS the
    model, so every key and shape must match exactly or loading raises
    -- no silent resizing."""

    def load(model: M) -> M:
        model.load_state_dict(state_dict(source, progress))
        return model

    return load
