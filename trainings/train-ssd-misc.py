"""Train the pure-SSD flavor on the misc drone dataset: background vs a
single collapsed "object" class (people, cars, buses, ...). All images
are 480p, trained and deployed at 480x640 -- no resizing anywhere.

Everything is stock building blocks wired together: skorch engine,
EpochScoring computes validation mAP each epoch, Checkpoint keeps the
best-mAP weights, MlflowLogger sends the history to MLflow.

The one deviation from the SSDLITE preset: anchor sizes. The preset's
smallest anchor is 32 px, and the matcher gives up below IoU 0.2, so
objects under ~15 px would never be matched at all. 16 px anchors on
the stride-16 level pull the floor down to roughly 12 px; below that
SSD at these strides is blind (a stride-8 level would need a model
change).

Run from the repo root:  python trainings/train-ssd-misc.py
Inspect runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db
(experiment "ssd-misc"; recent mlflow rejects the legacy ./mlruns file
store, hence sqlite -- set MLFLOW_TRACKING_URI to log elsewhere)
"""

import os
import random
from dataclasses import replace
from functools import partial
from pathlib import Path

import mlflow
from skorch.callbacks import Checkpoint, EpochScoring, MlflowLogger

from modelinhos.engine.skorch import skorch_engine
from modelinhos.evaluation import mean_average_precision
from modelinhos.models.anchors import anchors
from modelinhos.models.ssdlite import SSDLITE
from modelinhos.preprocess.lables import LabelEncoder
from modelinhos.sample import Annotation, Sample, read_samples
from modelinhos.zoo import build_ssd


def main(
    path: Path = Path("/mnt/data/misc/annotations.json"),
    resolution: tuple[int, int] = (480, 640),  # height, width
    checkpoints: Path = Path("trainings/checkpoints/ssd-misc"),
    max_epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 32,
    num_workers: int = 8,
    val_fraction: float = 0.15,
    seed: int = 137,
    device: str = "cuda",
):
    # Background vs anything: collapse every label into one class.
    samples = [
        Sample(
            file_name=sample.file_name,
            annotations=[
                Annotation(bbox=ann.bbox, label="object", score=ann.score)
                for ann in sample.annotations
            ],
        )
        for sample in read_samples(path)
    ]

    # NB: if the images are consecutive frames of the same flight, a
    # random split puts near-duplicates on both sides and inflates val
    # mAP -- group by flight/sequence instead then.
    random.Random(seed).shuffle(samples)
    n_val = int(len(samples) * val_fraction)
    val, train = samples[:n_val], samples[n_val:]
    print(f"train: {len(train)} samples, val: {len(val)} samples")

    lencoder = LabelEncoder(
        resolution=resolution,
        l2i={"__background__": 0, "object": 1},
    )

    def val_map(net, X, y=None) -> float:
        y_pred = lencoder.inverse_transform(net.predict(X))
        y_true = lencoder.inverse_transform(X.samples)
        result = mean_average_precision(y_true, y_pred, l2i=lencoder.l2i)
        return float(result["mAP"].iloc[0])

    checkpoint = Checkpoint(
        monitor="valid_mAP_best",
        dirname=str(checkpoints),
        f_optimizer=None,
        f_criterion=None,
    )

    detector = build_ssd(
        resolution,
        lencoder=lencoder,
        arch=replace(
            SSDLITE,
            anchors=partial(
                anchors,
                sizes=[[16, 32], [64, 128], [128, 256]],
                steps=[16, 32, 64],
            ),
        ),
        engine=skorch_engine(
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
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
                MlflowLogger(terminate_after_train=False),
            ],
        ),
    )

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    mlflow.set_experiment("ssd-misc")
    with mlflow.start_run():
        detector.fit(train, val_samples=val)

    y_pred = detector.transform(val)
    final = mean_average_precision(val, y_pred, l2i=lencoder.l2i)
    print(f"final val mAP: {final['mAP'].iloc[0]:.4f}")
    print(f"best weights: {checkpoints / checkpoint.f_params}")


if __name__ == "__main__":
    main()
