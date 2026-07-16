from collections.abc import Callable
from dataclasses import fields
from functools import partial
from typing import Tuple

import torch
from torch import nn

from modelinhos.loss.matching import match
from modelinhos.loss.subloss import Sublosses, WeightedLoss
from modelinhos.tasks.standard import (
    PerBatch,
    PerBatchEncoded,
    StandardDetection,
    map_fields,
)


def select(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    anchors: torch.Tensor,
    use_negatives: bool,
    positives: torch.Tensor,
    negatives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b_pos, a_pos, o_pos = torch.where(positives)
    pred_pos = y_pred[b_pos, a_pos]
    true_pos = y_true[b_pos, o_pos]
    anch_pos = anchors[a_pos]

    if not use_negatives:
        return pred_pos, true_pos, anch_pos

    b_neg, a_neg = torch.where(negatives)
    pred_neg = y_pred[b_neg, a_neg]
    true_neg = torch.zeros_like(pred_neg[:, 0], dtype=torch.long)
    anch_neg = anchors[a_neg]

    pred_all = torch.cat([pred_pos, pred_neg], dim=0)
    true_all = torch.cat([true_pos.view(-1), true_neg], dim=0).long()
    anch_all = torch.cat([anch_pos, anch_neg], dim=0)
    return pred_all, true_all, anch_all


Matching = Callable[
    [
        StandardDetection[torch.Tensor],
        StandardDetection[torch.Tensor],
        torch.Tensor,
    ],
    Tuple[torch.Tensor, torch.Tensor],
]


class DetectionLoss(nn.Module):
    def __init__(
        self,
        priors: torch.Tensor,
        sublosses: Sublosses,
        match: Matching = partial(
            match,
            negpos_ratio=7,
            overalp=0.35,
        ),
    ) -> None:
        super().__init__()
        if not isinstance(sublosses, StandardDetection):
            raise TypeError(
                "sublosses must be a StandardDetection[WeightedLoss]"
            )
        self.sublosses = sublosses
        self.match = match
        self.register_buffer("priors", priors)

    def forward(
        self,
        y_true: StandardDetection[torch.Tensor],
        y_pred: StandardDetection[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        positives, negatives = self.match(
            y_pred,
            y_true,
            self.priors,
        )

        losses = {}
        for field in fields(self.sublosses):
            name = field.name
            subloss: WeightedLoss = getattr(self.sublosses, name)
            if subloss.loss is None:
                continue
            y_pred_, y_true_, anchor_ = select(
                getattr(y_pred, name),
                getattr(y_true, name),
                self.priors,
                use_negatives=subloss.needs_negatives,
                positives=positives,
                negatives=negatives,
            )
            losses[name] = subloss(y_pred_, y_true_, anchor_)

        losses["loss"] = torch.stack(tuple(losses.values())).sum()
        return losses

    def decode(self, predictions: PerBatchEncoded) -> PerBatch:
        """Apply each subloss's dec_pred to its field of the raw model
        output -- the loss owns the codec, so decoding lives with it."""
        return map_fields(
            lambda subloss, pred: subloss.dec_pred(pred),
            self.sublosses,
            predictions,
            into=PerBatch,
        )
