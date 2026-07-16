from dataclasses import dataclass
from typing import Callable, Optional

import torch
import tqdm

from modelinhos.evaluation import MetricCollector

DLBuilder = Callable[
    [torch.utils.data.Dataset, Callable],
    torch.utils.data.DataLoader,
]


def default_dataloader_builder(
    dataset,
    collate_fn,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_fn,
    )


def default_optimizer_builder(
    params,
    lr: float = 1e-3,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=lr)


@dataclass(frozen=True)
class TrainConfig:
    """Architecture-independent optimization knobs -- everything about how
    to run the optimization, nothing about what is being optimized."""

    epochs: int = 1
    lr: float = 1e-3
    optimizer_builder: Callable = default_optimizer_builder
    metrics: Optional[Callable] = None
    train_dataloader_builder: DLBuilder = default_dataloader_builder
    valid_dataloader_builder: DLBuilder = default_dataloader_builder
    device: Optional[str] = None


class SimpleTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        decode: Callable,
        collate,
        label_encoder,
        loss_fn: Callable = print,
        config: TrainConfig = TrainConfig(),
    ):
        self.decode = decode
        self.collate = collate
        self.lencoder = label_encoder
        self.metrics_fn = config.metrics or MetricCollector
        self.train_dataloader_builder = config.train_dataloader_builder
        self.valid_dataloader_builder = config.valid_dataloader_builder
        self.epochs = config.epochs

        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        # loss_fn may register buffers (e.g. DetectionLoss.priors) that need
        # to live on the same device as the model/data; plain callables
        # (e.g. the `print` default) have no `.to()` to call.
        self.loss_fn = (
            loss_fn.to(self.device)
            if isinstance(loss_fn, torch.nn.Module)
            else loss_fn
        )
        self.optimizer = config.optimizer_builder(
            self.model.parameters(), config.lr
        )

    def _score(self, metric_fn, batch, preds):
        true = self.lencoder.inverse_transform(
            self.collate.un_batch(batch),
        )
        pred = self.lencoder.inverse_transform(
            self.collate.un_batch_nms(self.decode(preds))
        )
        metric_fn(true, pred)

    def fit(self, dataset, val_dataset=None) -> "SimpleTrainer":
        loader = self.train_dataloader_builder(dataset, self.collate.collate)

        for epoch in range(self.epochs):
            self.model.train()
            trainm = self.metrics_fn(self.lencoder.l2i)
            epoch_loss, n_batches = 0.0, 0
            for images, batch in tqdm.tqdm(loader, desc=f"Epoch {epoch}"):
                images = images.to(self.device)
                batch = batch.to(self.device)
                preds = self.model(images)
                loss = self.loss_fn(batch, preds)
                loss = loss["loss"] if isinstance(loss, dict) else loss
                print(loss)

                self.optimizer.zero_grad()
                if torch.is_tensor(loss):
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                self._score(trainm, batch, preds)

            if n_batches:
                print(f"Epoch {epoch}, train loss: {epoch_loss / n_batches}")
            print(f"Epoch {epoch}, train mAP: {trainm.value().iloc[0]['mAP']}")

            if val_dataset is None:
                continue
            validm = self.metrics_fn(self.lencoder.l2i)
            self._validate(validm, val_dataset)
            print(f"Epoch {epoch}, valid mAP: {validm.value().iloc[0]['mAP']}")

        return self

    def _validate(self, metrics_fn, dataset) -> None:
        loader = self.valid_dataloader_builder(dataset, self.collate.collate)
        self.model.eval()
        with torch.no_grad():
            for images, batch in tqdm.tqdm(loader, desc="Validation"):
                images = images.to(self.device)
                batch = batch.to(self.device)
                preds = self.model(images)
                self.loss_fn(batch, preds)
                self._score(metrics_fn, batch, preds)

    def predict(self, dataset) -> list:
        loader = self.valid_dataloader_builder(dataset, self.collate.collate)
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, _ in tqdm.tqdm(loader, desc="Prediction"):
                images = images.to(self.device)
                results.extend(
                    self.collate.un_batch_nms(self.decode(self.model(images)))
                )
        return results

    def predict_single(self, blob: torch.Tensor) -> list:
        self.model.eval()
        with torch.no_grad():
            preds = self.collate.un_batch_nms(
                self.decode(self.model(blob.to(self.device)))
            )
        return preds
