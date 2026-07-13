import functools
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch

LossFunctionyType = Union[
    torch.nn.Module,
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
]


@dataclass
class WeightedLoss:
    loss: Optional[LossFunctionyType]
    weight: float = 1.0
    dec_pred: Callable = lambda x, _: x
    enc_pred: Callable = lambda x, _: x
    enc_true: Callable = lambda x, _: x
    needs_negatives: bool = False

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
