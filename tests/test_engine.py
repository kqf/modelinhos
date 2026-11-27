import pathlib
from collections.abc import Callable

import matplotlib
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from skorch.callbacks import Checkpoint

from modelinhos.detector import build_detector
from modelinhos.engine.lit import lightning_engine
from modelinhos.engine.simple import simple_engine
from modelinhos.engine.skorch import skorch_engine
from modelinhos.evaluation import mean_average_precision
from modelinhos.models.blazenet import RETINANET_F
from modelinhos.models.load import restore, warm_start
from modelinhos.preprocess.labels import LabelEncoder
from modelinhos.sample import read_samples


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param(simple_engine(max_epochs=1), id="simple"),
        pytest.param(skorch_engine(max_epochs=1), id="skorch"),
        pytest.param(lightning_engine(max_epochs=1), id="lightning"),
    ],
)
def test_engine(
    engine: Callable,
    resolution: tuple[int, int],
    dataset: pathlib.Path,
    tmp_path: pathlib.Path,
):

    matplotlib.use("Agg")
    data = read_samples(dataset)

    lencoder = LabelEncoder(
        l2i={"__background__": 0, "dot": 1},
    ).fit(data)

    # It's the smallest one, check it overfits
    model = build_detector(
        resolution=resolution,
        lencoder=lencoder,
        arch=RETINANET_F,
        weights=warm_start(RETINANET_F.pretraining, fresh=RETINANET_F.diff),
        engine=engine,
    )
    model.fit(data)
    y_pred = model.transform(data)
    m_ap = mean_average_precision(
        data,
        y_pred,
        model.label_encoder.l2i,
        resolution=resolution,
    )
    assert m_ap["mAP"].iloc[0] == pytest.approx(1.0)

    # Every engine writes the same plain .pt: raw-model keys, nothing
    # else -- so a strict restore rebuilds the exact trained detector.
    model.save(tmp_path / "weights.pt")
    state = torch.load(tmp_path / "weights.pt", weights_only=True)
    assert not any(name.startswith("model.") for name in state)

    restored = build_detector(
        resolution=resolution,
        lencoder=lencoder,
        arch=RETINANET_F,
        weights=restore(tmp_path / "weights.pt"),
        engine=simple_engine(),
    )
    # The engine may have predicted on another device -- compare with a
    # tolerance rather than bit-for-bit
    for y, z in zip(y_pred, restored.transform(data), strict=True):
        assert [a.label for a in y.annotations] == [
            a.label for a in z.annotations
        ]
        for a, b in zip(y.annotations, z.annotations, strict=True):
            assert a.bbox == pytest.approx(b.bbox, abs=1e-4)
            assert a.score == pytest.approx(b.score, abs=1e-4)


def skorch_checkpointed(
    dirpath: pathlib.Path,
) -> tuple[Callable, pathlib.Path]:
    # Stock Checkpoint, weights only: f_params holds the DetectionModel
    # wrapper's dict, whose model. prefix restore() strips
    checkpoint = Checkpoint(
        dirname=str(dirpath),
        f_params="params.pt",
        f_optimizer=None,  # type: ignore
        f_criterion=None,  # type: ignore
        f_history=None,  # type: ignore
    )
    engine = skorch_engine(max_epochs=1, callbacks=[checkpoint])
    return engine, dirpath / "params.pt"


def lightning_checkpointed(
    dirpath: pathlib.Path,
) -> tuple[Callable, pathlib.Path]:
    # Stock ModelCheckpoint, weights only: the .ckpt nests LitDetection's
    # state dict, which restore() unwraps
    checkpoint = ModelCheckpoint(
        dirpath=dirpath,
        filename="params",
        monitor="val_loss",
        save_weights_only=True,
    )
    engine = lightning_engine(
        max_epochs=1,
        trainer_kwargs={"callbacks": [checkpoint]},
    )
    return engine, dirpath / "params.ckpt"


@pytest.mark.parametrize(
    "checkpointed",
    [
        pytest.param(skorch_checkpointed, id="skorch"),
        pytest.param(lightning_checkpointed, id="lightning"),
    ],
)
def test_best_epoch_checkpoint_restores(
    checkpointed: Callable,
    resolution: tuple[int, int],
    dataset: pathlib.Path,
    tmp_path: pathlib.Path,
):
    data = read_samples(dataset)
    lencoder = LabelEncoder(
        l2i={"__background__": 0, "dot": 1},
    ).fit(data)
    engine, path = checkpointed(tmp_path / "checkpoints")

    # The callbacks monitor a validation metric, so pass a val split
    build_detector(
        resolution=resolution,
        lencoder=lencoder,
        arch=RETINANET_F,
        weights=warm_start(RETINANET_F.pretraining, fresh=RETINANET_F.diff),
        engine=engine,
    ).fit(data, data)

    # The strict restore accepts the checkpoint as written
    build_detector(
        resolution=resolution,
        lencoder=lencoder,
        arch=RETINANET_F,
        weights=restore(path),
        engine=simple_engine(),
    )
