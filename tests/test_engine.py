import pathlib
from collections.abc import Callable

import matplotlib
import pytest

from modelinhos.engine.lit import lightning_engine
from modelinhos.engine.simple import simple_engine
from modelinhos.engine.skorch import skorch_engine
from modelinhos.evaluation import mean_average_precision
from modelinhos.models.blazenet import RETINANET_F
from modelinhos.preprocess.labels import LabelEncoder
from modelinhos.sample import read_samples
from modelinhos.zoo import build_blaze


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
):

    matplotlib.use("Agg")
    data = read_samples(dataset)

    lencoder = LabelEncoder(
        l2i={"__background__": 0, "dot": 1},
    ).fit(data)

    # It's the smallest one, check it overfits
    model = build_blaze(
        arch=RETINANET_F,
        resolution=resolution,
        lencoder=lencoder,
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
