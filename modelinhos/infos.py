"""Model-facing facts and verdicts. The rule for the split with
modelinhos.analysis: if the answer changes when you swap the model, it
lives here; data-only questions stay in analysis."""

import statistics
import time

import numpy as np
import pandas as pd
import torch
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from modelinhos.containers import Collate
from modelinhos.data import SampleDataset
from modelinhos.detector import DetectionRecipe
from modelinhos.preprocess.lables import SampleEncoder
from modelinhos.sample import Annotation, Sample
from modelinhos.tasks.standard import PerBatchEncoded


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


def matchability(
    samples: list[Sample[Annotation]],
    recipe: DetectionRecipe,
    resolution: tuple[int, int],
    lencoder: SampleEncoder,
) -> pd.DataFrame:
    """Fact: the recipe's own matcher simulated over every GT box --
    priors from recipe.anchors, positives and negatives from the
    matcher recipe.loss binds, so there is no matching knob here to
    misconfigure. One row per box: relative geometry (w, h), scale in
    pixels at the model resolution, and matched -- how many anchors
    the matcher would hand this box at training time. matched == 0 is
    a box the loss can never see: the exact recall ceiling lives in
    this column. negatives is an image-level count repeated on the
    image's rows, and only the count is a fact -- which anchors get
    collected depends on live predictions (dummy logits here); images
    without boxes contribute no rows, so their negatives (legitimate
    background) stay out of the frame. y_true rides the training
    pipeline itself -- lencoder.transform -> SampleDataset -> DataLoader
    -> Collate, with a blank frame injected via read_image instead of
    disk I/O (encoded by the recipe's own iencoder, so the image leg of
    the contract stays exercised too) -- shapes, dtypes and empty-image
    quirks are training's, not this function's. lencoder must cover the
    samples' labels: the encoded indices reach the matcher's mining
    cross-entropy, which is also why the dummy logits are n_classes
    wide."""
    H, W = resolution
    priors = recipe.anchors((H, W))
    loss = recipe.loss(priors=priors, score_thresh=0.4)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    loader = DataLoader(
        SampleDataset(
            lencoder.transform(samples),
            encode_images=recipe.iencoder,
            read_image=lambda file_name: frame,
        ),
        batch_size=1,
        collate_fn=Collate().collate,
    )
    rows = []
    for sample, (_, y_true) in zip(samples, loader):
        positives, negatives = loss.match(
            PerBatchEncoded(
                bboxes=torch.zeros(1, len(priors), 4),
                scores=torch.zeros(1, len(priors), lencoder.n_classes),
                labels=torch.zeros(1, len(priors), lencoder.n_classes),
            ),
            y_true,
            priors,
        )
        counts = positives[0].sum(dim=0)
        mined = int(negatives[0].sum())
        for annotation, count in zip(sample.annotations, counts.tolist()):
            w = annotation.bbox[2] - annotation.bbox[0]
            h = annotation.bbox[3] - annotation.bbox[1]
            rows.append(
                {
                    "file": str(sample.file_name),
                    "label": annotation.label,
                    "w": w,
                    "h": h,
                    "scale": (w * W * h * H) ** 0.5,
                    "matched": int(count),
                    "negatives": mined,
                }
            )
    return pd.DataFrame(rows)


def anchor_advice(
    matched: pd.DataFrame,
    recipe: DetectionRecipe,
    resolution: tuple[int, int],
) -> pd.DataFrame:
    """Verdict: recall ceilings from the matchability fact, one row
    per label and size octave (bucket is the octave's lower edge in
    pixels; "split" is honoured as a grouping column when present).
    matchable is the share of boxes the matcher can see at all -- no
    training run beats it. The serviceable range is the recipe's own
    anchor scale bracket, and the advice column names the knob: below
    the floor only resolution helps (steps come from the backbone,
    sizes from data), above the ceiling coarser sizes or a lower
    resolution; in-range misses point at overlap / anchor layout."""
    H, W = resolution
    priors = recipe.anchors((H, W))
    scales = (priors[:, 2] * W * priors[:, 3] * H).sqrt()
    floor, ceiling = float(scales.min()), float(scales.max())
    keys = [key for key in ("split", "label") if key in matched.columns]
    view = matched.assign(
        bucket=2.0 ** np.floor(np.log2(matched.scale)),
        matchable=matched.matched > 0,
        below=matched.scale < floor,
        above=matched.scale > ceiling,
    )
    ceilings = view.groupby(keys + ["bucket"], as_index=False).agg(
        boxes=("matchable", "size"),
        matchable=("matchable", "mean"),
        anchors_per_box=("matched", "mean"),
        below_floor=("below", "mean"),
        above_ceiling=("above", "mean"),
    )
    return ceilings.assign(
        advice=np.select(
            [
                ceilings.below_floor > 0,
                ceilings.above_ceiling > 0,
                ceilings.matchable < 1,
            ],
            [
                f"below the {floor:.0f}px anchor floor: "
                "raise the resolution",
                f"above the {ceiling:.0f}px anchor ceiling: "
                "add coarser sizes or lower the resolution",
                "in range but unmatched: lower overlap or add "
                "sizes/ratios near this octave",
            ],
            default="",
        )
    )
