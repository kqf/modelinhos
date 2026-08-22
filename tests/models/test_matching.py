from dataclasses import asdict
from functools import partial

import pytest
import torch

from modelinhos.loss.loss import DetectionLoss
from modelinhos.loss.matching import atss_boxes, atss_match, match
from modelinhos.models.anchors import AnchorConfig, anchors, level_sizes
from modelinhos.models.ssdlite import build_ssd_loss
from modelinhos.tasks.standard import PerBatch, PerBatchEncoded

SSDLITE_ANCHOR_CONFIG = AnchorConfig(
    sizes=[(32, 64), (64, 128), (128, 256)],
    steps=[16, 32, 64],
)


@pytest.fixture
def resolution() -> tuple[int, int]:
    # Overrides the 800x1088 shared fixture -- matching doesn't need a
    # real training resolution and small grids keep the test readable.
    return 128, 128


@pytest.fixture
def priors(resolution) -> torch.Tensor:
    return anchors(resolution, **asdict(SSDLITE_ANCHOR_CONFIG), clip=False)


@pytest.fixture
def levels(resolution) -> list[int]:
    return level_sizes(resolution, **asdict(SSDLITE_ANCHOR_CONFIG))


@pytest.fixture
def y_true() -> PerBatch:
    # One real box centred on the image plus one padding row, exactly as
    # collate_labels pads ragged batches (pad_value=-1) -- label ids are
    # int64 there, whether or not the batch had any annotations.
    labels = torch.tensor([[[1], [-1]]])
    return PerBatch(
        bboxes=torch.tensor(
            [
                [
                    [0.25, 0.25, 0.75, 0.75],
                    [-1.0, -1.0, -1.0, -1.0],
                ]
            ]
        ),
        scores=torch.ones_like(labels, dtype=torch.float32),
        labels=labels,
    )


@pytest.fixture
def y_pred(priors, n_classes: int = 2) -> PerBatchEncoded:
    torch.manual_seed(137)
    logits = torch.randn(1, priors.shape[0], n_classes)
    return PerBatchEncoded(
        bboxes=torch.randn(1, priors.shape[0], 4),
        scores=logits,
        labels=logits,
    )


def test_level_sizes_match_anchor_layout():
    for resolution in [(128, 128), (96, 128)]:
        for aspect_ratios in [None, (0.5, 1.0, 2.0)]:
            expected = anchors(
                resolution,
                **asdict(SSDLITE_ANCHOR_CONFIG),
                aspect_ratios=aspect_ratios,
            ).shape[0]
            sizes = level_sizes(
                resolution,
                **asdict(SSDLITE_ANCHOR_CONFIG),
                aspect_ratios=aspect_ratios,
            )
            assert sum(sizes) == expected


def test_atss_is_drop_in_for_match(y_pred, y_true, priors, levels):
    # Both matchers reduce to the same (y_pred, y_true, anchors) tail
    # once their knobs are partial'd away -- what DetectionLoss calls.
    reference = partial(match, negpos_ratio=7, overalp=0.35)
    candidate = partial(atss_match, negpos_ratio=7, level_sizes=levels)

    expected_pos, expected_neg = reference(y_pred, y_true, priors)
    positives, negatives = candidate(y_pred, y_true, priors)

    assert positives.shape == expected_pos.shape
    assert positives.dtype == expected_pos.dtype
    assert negatives.shape == expected_neg.shape
    assert negatives.dtype == expected_neg.dtype


def test_atss_assigns_centred_gt(y_pred, y_true, priors, levels):
    positives, negatives = atss_match(
        y_pred,
        y_true,
        priors,
        negpos_ratio=7,
        level_sizes=levels,
    )

    # The real GT gets anchors, and only anchors centred inside it
    matched = positives[0, :, 0]
    assert matched.any()
    centres = priors[matched, :2]
    assert (centres > 0.25).all()
    assert (centres < 0.75).all()

    # The padding row must claim nothing
    assert not positives[0, :, 1].any()

    # Negatives are mined outside the positives, respecting the ratio
    assert not (negatives[0] & positives[0].any(dim=-1)).any()
    assert negatives[0].sum() <= 7 * positives[0].sum()


def test_atss_gives_each_anchor_one_gt(y_pred, priors, levels):
    labels = torch.tensor([[[1], [1]]])
    y_true = PerBatch(
        bboxes=torch.tensor(
            [
                [
                    [0.25, 0.25, 0.75, 0.75],
                    [0.30, 0.30, 0.80, 0.80],
                ]
            ]
        ),
        scores=torch.ones_like(labels, dtype=torch.float32),
        labels=labels,
    )
    positives, _ = atss_match(
        y_pred,
        y_true,
        priors,
        negpos_ratio=7,
        level_sizes=levels,
    )
    assert (positives.sum(dim=-1) <= 1).all()


def test_atss_handles_empty_and_bad_levels(priors):
    empty = atss_boxes(torch.empty((0, 1)), priors, topk=9)
    assert empty.shape == (priors.shape[0], 0)

    with pytest.raises(ValueError):
        atss_boxes(
            torch.tensor([[0.2, 0.2, 0.6, 0.6]]),
            priors,
            topk=9,
            level_sizes=[1, 2, 3],
        )


def test_atss_plugs_into_detection_loss(y_pred, y_true, priors, levels):
    reference = build_ssd_loss(priors, score_thresh=0.5)
    loss_fn = DetectionLoss(
        priors=priors,
        sublosses=reference.sublosses,
        match=partial(atss_match, negpos_ratio=7, level_sizes=levels),
    )
    losses = loss_fn(y_pred, y_true)
    assert torch.isfinite(losses["loss"])
    assert losses["loss"] > 0
