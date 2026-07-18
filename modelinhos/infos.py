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
    counts, weight size and activation memory from torchinfo (the
    forward/backward total at the given shape -- with the weights, the
    peak-RAM side of the budget), forward FLOPs from torch's own
    counter. shape is the full input batch (n, channels, h, w); the
    model resolution is its last two dims. min/max_bbox_px bracket the
    recipe's anchor scales (sqrt of area, in pixels at that
    resolution): below the floor no box can be matched, far above the
    ceiling none either -- pure model properties. n_anchors is the
    prior count the head predicts and NMS consumes -- the decode-side
    cost the FLOPs of the backbone do not show. Latency is the only
    in-house measurement:
    median wall time at the given shape, eval mode, on the machine
    running this -- never a prediction for the target device (rerun
    there instead). The big counts come back as underscore-grouped
    strings (1_000_000) -- for reading, not arithmetic."""
    *_, H, W = shape
    model = recipe.build_model(resolution=(H, W), n_classes=n_classes)
    model.eval()
    priors = recipe.anchors((H, W))
    scales = (priors[:, 2] * W * priors[:, 3] * H).sqrt()
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
                "activations_mb": info.total_output_bytes / 2**20,
                "flops": f"{counter.get_total_flops():_}",
                "n_anchors": f"{len(priors):_}",
                "min_bbox_px": float(scales.min()),
                "max_bbox_px": float(scales.max()),
                "latency_ms": 1e3 * statistics.median(timings),
            }
        ]
    )
