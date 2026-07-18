"""Model-facing facts and verdicts. The rule for the split with
modelinhos.analysis: if the answer changes when you swap the model, it
lives here; data-only questions stay in analysis."""

import statistics
import time

import pandas as pd
import torch
import torchinfo
from torch.utils.flop_counter import FlopCounterMode

from modelinhos.detector import DetectionRecipe


def summarize(
    recipe: DetectionRecipe,
    shape: tuple[int, int, int, int],
    n_classes: int,
    warmup: int = 5,
    repeats: int = 20,
) -> pd.DataFrame:
    """Fact: one row of data-independent model numbers -- parameter
    counts and weight size from torchinfo, forward FLOPs from torch's
    own counter. shape is the full input batch (n, channels, h, w); the
    model resolution is its last two dims. min_bbox_px is the scale
    (sqrt of area, in pixels at that resolution) of the recipe's
    smallest anchor -- the floor below which no box can be matched, a
    pure model property. Latency is the only in-house measurement:
    median wall time at the given shape, eval mode, on the machine
    running this -- never a prediction for the target device (rerun
    there instead). The big counts come back as underscore-grouped
    strings (1_000_000) -- for reading, not arithmetic."""
    *_, H, W = shape
    model = recipe.build_model(resolution=(H, W), n_classes=n_classes)
    model.eval()
    priors = recipe.anchors((H, W))
    image = torch.rand(*shape)
    info = torchinfo.summary(model, input_data=image, verbose=0)
    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            model(image)
        for _ in range(warmup):
            model(image)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(image)
            timings.append(time.perf_counter() - start)
    return pd.DataFrame(
        [
            {
                "params": f"{info.total_params:_}",
                "trainable": f"{info.trainable_params:_}",
                "size_mb": info.total_param_bytes / 2**20,
                "flops": f"{counter.get_total_flops():_}",
                "min_bbox_px": float(
                    (priors[:, 2] * W * priors[:, 3] * H).sqrt().min()
                ),
                "latency_ms": 1e3 * statistics.median(timings),
            }
        ]
    )
