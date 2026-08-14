import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import torch

from modelinhos.tasks.standard import StandardDetection

LossFunctionyType = Union[
    torch.nn.Module,
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]


@dataclass
class WeightedLoss:
    loss: LossFunctionyType | None
    weight: float = 1.0
    dec_pred: Callable = lambda x: x
    enc_pred: Callable = lambda x, _: x
    enc_true: Callable = lambda x, _: x
    needs_negatives: bool = False
    # Which y_true field this subloss trains against; None means its own
    # name. For heads whose target is derived from another field rather
    # than stored in the annotations -- e.g. an FCOS-style centerness
    # head lives in the scores slot but its target is a function of the
    # matched GT box (true_field="bboxes") and the anchor, computed by
    # enc_true.
    true_field: str | None = None

    def __call__(self, y_pred, y_true, anchors):
        y_pred_encoded = self.enc_pred(y_pred, anchors)
        y_true_encoded = self.enc_true(y_true, anchors)
        return self.weight * self.loss(y_pred_encoded, y_true_encoded)


def masked_loss(loss_function: LossFunctionyType) -> LossFunctionyType:
    @functools.wraps(loss_function)
    def f(pred: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(data)
        data_masked = data[mask]
        pred_masked = pred[mask]
        loss = loss_function(data_masked, pred_masked)
        if data_masked.numel() == 0:
            loss = torch.nan_to_num(loss, 0)
        return loss / max(data_masked.shape[0], 1)

    return f


def sum_normalized(loss_function: LossFunctionyType) -> LossFunctionyType:
    @functools.wraps(loss_function)
    def f(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        loss = loss_function(pred, true)
        if pred.shape[0] == 0:
            return torch.nan_to_num(loss, 0)
        return loss / pred.shape[0]

    return f


def positive_normalized(loss_function: LossFunctionyType) -> LossFunctionyType:
    """Sum-reduced loss divided by the number of positive targets
    (target > 0, background being 0). This is the SSD convention for the
    confidence loss: it is computed over positives plus mined negatives,
    but normalized by the positive count only -- dividing by the full
    count would underweight it ~(1 + negpos_ratio)x against the box loss,
    which is normalized by the same positive count."""

    @functools.wraps(loss_function)
    def f(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        loss = loss_function(pred, true)
        if pred.shape[0] == 0:
            return torch.nan_to_num(loss, 0)
        return loss / (true > 0).sum().clamp(min=1)

    return f


@dataclass(frozen=True)
class Sublosses(StandardDetection[WeightedLoss]):
    """Per-field losses of the standard detection task: for each of
    bboxes/scores/labels, the loss plus its encode/decode codecs."""


def retina_confidence_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> tuple[torch.Tensor]:
    n_pos = (y_true > 0).sum()
    loss = torch.nn.functional.cross_entropy(
        y_pred,
        y_true.view(-1),
        reduction="sum",
    )
    return loss / n_pos
