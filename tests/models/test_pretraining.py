import pytest

from modelinhos.detector import DetectionRecipe
from modelinhos.models.blazenet import (
    BLAZEFACE_B,
    BLAZEFACE_F,
    RETINANET_B,
    RETINANET_F,
)
from modelinhos.models.fcos import FCOS_DELTA, TORCHVISION_FCOS
from modelinhos.models.load import restore, state_dict, warm_start
from modelinhos.models.retinanet import RETINANET, TORCHVISION_RETINANET
from modelinhos.models.ssdlite import (
    SSDLARGE,
    SSDLITE,
    TORCHVISION_SSDLITE,
)


def build(arch: DetectionRecipe):
    # Size the model for the checkpoint's own label space, so a head
    # that mirrors the checkpoint layout matches shape for shape
    n_classes = len(arch.pretraining.meta["categories"])
    return arch.build_model(resolution=(320, 320), n_classes=n_classes)


def signature(state: dict) -> set[tuple[str, tuple[int, ...]]]:
    return {
        (name, tuple(param.shape))
        for name, param in state.items()
        # BatchNorm fills in the counter when a checkpoint lacks it
        if not name.endswith("num_batches_tracked")
    }


@pytest.mark.parametrize(
    "arch",
    [
        pytest.param(TORCHVISION_RETINANET, id="torchvision_retinanet"),
        pytest.param(TORCHVISION_SSDLITE, id="torchvision_ssdlite"),
        pytest.param(TORCHVISION_FCOS, id="torchvision_fcos"),
        pytest.param(BLAZEFACE_F, id="blazeface_f"),
        pytest.param(BLAZEFACE_B, id="blazeface_b"),
    ],
)
def test_pretraining_restores_strictly(arch: DetectionRecipe):
    restore(arch.pretraining, progress=False)(build(arch))


@pytest.mark.parametrize(
    "arch, new",
    [
        pytest.param(RETINANET, ("head.",), id="retinanet"),
        pytest.param(SSDLITE, ("head.",), id="ssdlite"),
        pytest.param(FCOS_DELTA, ("head.",), id="fcos_delta"),
        pytest.param(
            SSDLARGE,
            ("head.",),
            id="ssdlarge",
            marks=pytest.mark.xfail(
                strict=True,
                reason="backbone.features is flat, the checkpoint nests it"
                " one level deeper (features.0.0.0...), so warm_start"
                " loads almost nothing",
            ),
        ),
        pytest.param(
            RETINANET_F,
            # BlazeFace-front stops at stride 16: the stride-32 stage
            # has no counterpart in the checkpoint
            ("classifiers.", "regressors.", "backbone3."),
            id="retinanet_f",
        ),
        pytest.param(
            RETINANET_B,
            ("classifiers.", "regressors."),
            id="retinanet_b",
        ),
    ],
)
def test_pretraining_fits_all_but_heads(
    arch: DetectionRecipe,
    new: tuple[str, ...],
):
    model = build(arch)
    checkpoint = state_dict(arch.pretraining, progress=False)

    unmatched = {
        (name, shape)
        for name, shape in signature(model.state_dict())
        - signature(checkpoint)
        if not name.startswith(new)
    }
    assert not unmatched
    warm_start(arch.pretraining, progress=False)(model)
