from modelinhos.detector import build_detector
from modelinhos.engine.simple import simple_engine
from modelinhos.models.blazenet import RETINANET_F
from modelinhos.models.load import from_scratch
from modelinhos.preprocess.labels import LabelEncoder
from modelinhos.sample import Sample


def test_transform_predicts_on_unlabelled_images(
    data: list[Sample],
    resolution: tuple[int, int],
):
    detector = build_detector(
        resolution=resolution,
        lencoder=LabelEncoder(l2i={"__background__": 0, "dot": 1}),
        arch=RETINANET_F,
        engine=simple_engine(),
        weights=from_scratch,
    )
    unlabelled: list[Sample] = [
        Sample(file_name=s.file_name, annotations=[]) for s in data
    ]
    assert len(detector.transform(unlabelled)) == len(unlabelled)
