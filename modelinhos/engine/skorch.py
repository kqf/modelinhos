"""skorch adapter -- the organizing example of an Engine.

skorch's own answer to "where does decode live?" is a NeuralNet
subclass that is allowed to know its criterion (NeuralNetClassifier
does exactly this for the CrossEntropyLoss/softmax pair). DetectionLoss
already owns the codec, so DetectionNet just delegates to
criterion_.decode. Importing this module requires skorch
(pip install modelinhos[skorch])."""

from typing import Callable

import torch

try:
    from skorch import NeuralNet
    from skorch.helper import predefined_split
except ImportError as e:
    raise ImportError(
        "the skorch engine needs skorch installed -- "
        "pip install modelinhos[skorch]"
    ) from e

from modelinhos.containers import Collate
from modelinhos.detector import Baked


class DetectionNet(NeuralNet):
    """sklearn-style detection estimator: fit(dataset) trains,
    predict(dataset) returns decoded, NMS'd samples in the model's
    normalized space."""

    def __init__(self, *args, collate: Collate, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate = collate

    def get_loss(self, y_pred, y_true, X=None, training=False):
        # skorch's default runs to_tensor(y_true), which rejects the
        # PerBatch dataclass -- it has .to(), so move it ourselves.
        loss = self.criterion_(y_pred, y_true.to(self.device))
        return loss["loss"] if isinstance(loss, dict) else loss

    def predict(self, X) -> list:
        return [
            sample
            for preds in self.forward_iter(X, training=False)
            for sample in self.collate.un_batch_nms(
                self.criterion_.decode(preds)
            )
        ]

    def predict_single(self, blob: torch.Tensor) -> list:
        self.module_.eval()
        with torch.no_grad():
            preds = self.criterion_.decode(self.module_(blob.to(self.device)))
        return self.collate.un_batch_nms(preds)


class SkorchEngine:
    """Maps the Engine protocol onto a DetectionNet: fit(dataset,
    val_dataset) becomes net.fit(dataset, y=None) with a predefined
    validation split."""

    def __init__(self, net: DetectionNet):
        self.net = net

    def fit(self, dataset, val_dataset=None) -> "SkorchEngine":
        if val_dataset is not None:
            self.net.set_params(train_split=predefined_split(val_dataset))
        self.net.fit(dataset, y=None)
        return self

    def predict(self, dataset) -> list:
        return self.net.predict(dataset)

    def predict_single(self, blob: torch.Tensor) -> list:
        return self.net.predict_single(blob)


def skorch_engine(
    max_epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 2,
    num_workers: int = 0,
    optimizer=torch.optim.Adam,
    **net_kwargs,
) -> Callable[[Baked], SkorchEngine]:
    """Baked -> SkorchEngine builder for build_detector(engine=...).
    batch_size/num_workers cover the common case (same vocabulary as the
    other engines); any extra net_kwargs go straight to DetectionNet in
    skorch's own vocabulary (callbacks, iterator_train__shuffle, device,
    ...) and win over the knobs."""

    def build(baked: Baked) -> SkorchEngine:
        return SkorchEngine(
            DetectionNet(
                module=baked.model,
                criterion=baked.loss,
                collate=baked.collate,
                # base NeuralNet defaults to ValidSplit(5); our val split
                # arrives via fit(val_dataset) as a predefined_split
                train_split=None,
                **{
                    "max_epochs": max_epochs,
                    "lr": lr,
                    "batch_size": batch_size,
                    "optimizer": optimizer,
                    "iterator_train__collate_fn": baked.collate.collate,
                    "iterator_valid__collate_fn": baked.collate.collate,
                    "iterator_train__num_workers": num_workers,
                    "iterator_valid__num_workers": num_workers,
                    **net_kwargs,
                },
            )
        )

    return build
