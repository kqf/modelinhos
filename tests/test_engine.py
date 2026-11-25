import pathlib
from collections.abc import Callable

import matplotlib
import pytest
import torch

from modelinhos.detector import build_detector
from modelinhos.engine.lit import (
    WeightsCheckpoint as LitWeights,
    lightning_engine,
)
from modelinhos.engine.simple import simple_engine
from modelinhos.engine.skorch import (
    WeightsCheckpoint as SkorchWeights,
    skorch_engine,
)
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
    checkpoint = SkorchWeights(dirname=str(dirpath))
    engine = skorch_engine(max_epochs=1, callbacks=[checkpoint])
    return engine, checkpoint.weights_path


def lightning_checkpointed(
    dirpath: pathlib.Path,
) -> tuple[Callable, pathlib.Path]:
    checkpoint = LitWeights(dirpath=dirpath)
    engine = lightning_engine(
        max_epochs=1,
        trainer_kwargs={"callbacks": [checkpoint]},
    )
    return engine, checkpoint.path


@pytest.mark.parametrize(
    "checkpointed",
    [
        pytest.param(skorch_checkpointed, id="skorch"),
        pytest.param(lightning_checkpointed, id="lightning"),
    ],
)
def test_best_epoch_checkpoint_is_plain(
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

    # Same format as Engine.save: raw-model keys only, so the strict
    # restore accepts it
    state = torch.load(path, weights_only=True)
    assert not any(name.startswith("model.") for name in state)
    build_detector(
        resolution=resolution,
        lencoder=lencoder,
        arch=RETINANET_F,
        weights=restore(path),
        engine=simple_engine(),
    )
