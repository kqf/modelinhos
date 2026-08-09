"""Train the SSDLARGE or RETINANET flavor on COCO: full label space,
grayscale input without normalization, deliberately no augmentations --
this is the baseline the augmentation step will later be measured
against. Raw (0, 255) intensities go straight into the network; the
first BatchNorm learns the input statistics itself.

Both recipes share the same anchor structure (retina_anchors: sizes
16-512 over strides 8/16/32) and train at the same resolution, so a run
pair isolates the backbone/loss effect: MobileNetV3 + softmax SSD loss
(ssdlarge) vs ResNet50-FPN + sigmoid focal loss (retinanet). The
per-model hyperparameters live in CONFIGS: Adam at 1e-3 suits the small
warm-started mobilenet, the resnet needs the gentler 1e-4 to stay
stable under focal loss; batch size is a memory budget, not a tuning
knob. Everything else is shared and logged to MLflow as params.

The schedule is one built-in: OneCycleLR ramps up over the first
`warmup` fraction of steps (from max_lr/25), then cosine-anneals --
stepped per batch. EarlyStopping watches the same validation mAP as the
Checkpoint, so a stalled run exits after `patience` epochs with the
best weights already on disk. val_map additionally records the
precision/recall operating point at confidence 0.5 (IoU 0.5) into the
history, so MLflow tracks all three curves per epoch.

Data is the val2017-derived annotations (data/coco/annotations.json,
5k images) randomly split -- enough for first trainings; for the real
118k train set point `annotations` at a train2017 export
(modelinhos.coco.download(..., split="train2017"), ~19 GB).

Run from the repo root:  python trainings/train-coco.py [ssdlarge|retinanet]
Inspect runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db
(experiment "coco"; set MLFLOW_TRACKING_URI to log elsewhere)
"""

import math
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlflow
import torch
from skorch.callbacks import (
    Checkpoint,
    EarlyStopping,
    EpochScoring,
    LRScheduler,
    MlflowLogger,
)

from modelinhos.coco import load_samples
from modelinhos.engine.skorch import skorch_engine
from modelinhos.evaluation import mean_average_precision
from modelinhos.models.retinanet import RETINANET
from modelinhos.models.ssdlite import SSDLARGE
from modelinhos.preprocess.image import grayscale_image_encoder
from modelinhos.preprocess.labels import LabelEncoder
from modelinhos.zoo import build_retina, build_ssd

CONFIGS: dict[str, dict[str, Any]] = {
    "ssdlarge": {
        "build": build_ssd,
        "arch": SSDLARGE,
        "lr": 1e-3,
        "batch_size": 32,
    },
    "retinanet": {
        "build": build_retina,
        "arch": RETINANET,
        "lr": 1e-4,
        "batch_size": 16,
    },
}


def main(
    model: str = "ssdlarge",
    annotations: Path = Path("data/coco/annotations.json"),
    resolution: tuple[int, int] = (320, 320),  # height, width
    max_epochs: int = 60,
    warmup: float = 0.1,  # fraction of total steps spent ramping up
    patience: int = 10,
    val_fraction: float = 0.15,
    num_workers: int = 8,
    seed: int = 137,
    device: str = "cuda",
):
    config = CONFIGS[model]
    checkpoints = Path(f"trainings/checkpoints/coco-{model}")

    samples = load_samples(annotations)
    random.Random(seed).shuffle(samples)
    n_val = int(len(samples) * val_fraction)
    val, train = samples[:n_val], samples[n_val:]
    print(f"train: {len(train)} samples, val: {len(val)} samples")

    # Fit on everything before the split, so classes that ended up only
    # in val still get a head channel and count as misses instead of
    # raising a KeyError in transform().
    lencoder = LabelEncoder().fit(samples)
    val_classes = sorted(
        {lencoder.l2i[a.label] for sample in val for a in sample.annotations}
    )

    def val_map(net, X, y=None) -> float:
        y_pred = lencoder.inverse_transform(net.predict(X))
        y_true = lencoder.inverse_transform(X.samples)
        result = mean_average_precision(
            y_true, y_pred, l2i=lencoder.l2i, resolution=resolution
        )
        # The PR curve walks predictions by descending confidence, so
        # per class the last point still >= 0.5 is the operating point
        # at confidence 0.5. Macro averages: precision over the classes
        # the model actually predicted, recall over every class present
        # in the val ground truth (a class never predicted recalls 0).
        point = result[result["threshold"] >= 0.5].groupby("class_id").last()
        net.history.record(
            "valid_precision_at_50",
            float(point["precision"].mean()) if len(point) else 0.0,
        )
        net.history.record(
            "valid_recall_at_50",
            float(point["recall"].reindex(val_classes).fillna(0.0).mean()),
        )
        return float(result["mAP"].iloc[0])

    checkpoint = Checkpoint(
        monitor="valid_mAP_best",
        dirname=str(checkpoints),
        f_optimizer=None,
        f_criterion=None,
    )

    detector = config["build"](
        resolution,
        lencoder=lencoder,
        arch=replace(
            config["arch"],
            iencoder=grayscale_image_encoder(resolution),
        ),
        engine=skorch_engine(
            max_epochs=max_epochs,
            lr=config["lr"],
            batch_size=config["batch_size"],
            num_workers=num_workers,
            device=device,
            iterator_train__shuffle=True,
            callbacks=[
                EpochScoring(
                    val_map,
                    name="valid_mAP",
                    lower_is_better=False,
                    use_caching=False,
                ),
                checkpoint,
                EarlyStopping(
                    monitor="valid_mAP",
                    lower_is_better=False,
                    patience=patience,
                ),
                LRScheduler(
                    policy=torch.optim.lr_scheduler.OneCycleLR,
                    max_lr=config["lr"],
                    total_steps=max_epochs
                    * math.ceil(len(train) / config["batch_size"]),
                    pct_start=warmup,
                    step_every="batch",
                ),
                MlflowLogger(terminate_after_train=False),
            ],
        ),
    )

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment("coco")
    with mlflow.start_run(run_name=model):
        mlflow.log_params(
            {
                "model": model,
                "resolution": resolution,
                "lr": config["lr"],
                "batch_size": config["batch_size"],
                "max_epochs": max_epochs,
                "warmup": warmup,
                "patience": patience,
                "val_fraction": val_fraction,
                "seed": seed,
            }
        )
        detector.fit(train, val_samples=val)

    y_pred = detector.transform(val)
    final = mean_average_precision(
        val, y_pred, l2i=lencoder.l2i, resolution=resolution
    )
    print(f"final val mAP: {final['mAP'].iloc[0]:.4f}")
    print(f"best weights: {checkpoints / checkpoint.f_params}")


if __name__ == "__main__":
    main(*sys.argv[1:])  # type: ignore[arg-type]
