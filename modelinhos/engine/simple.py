"""Dependency-free reference implementation of the Engine protocol --
what the tests run against, and the template for writing a new adapter.
Metrics and label handling live outside the loop on purpose: evaluate
after fit via Detector.transform, so no engine is forced to know about
label encoders."""

from functools import partial
from typing import Callable, Optional

import torch
import tqdm

from modelinhos.detector import Baked

DLBuilder = Callable[
    [torch.utils.data.Dataset, Callable],
    torch.utils.data.DataLoader,
]


def default_dataloader_builder(
    dataset,
    collate_fn,
    shuffle: bool = False,
    batch_size: int = 2,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )


def default_optimizer_builder(
    params,
    lr: float = 1e-3,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=lr)


class SimpleTrainer:
    """Minimal training loop over Baked pieces. Everything
    detection-specific arrives pre-built (decode, collate); the loop
    itself only knows tensors."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        decode: Callable,
        collate,
        max_epochs: int = 1,
        lr: float = 1e-3,
        optimizer_builder: Callable = default_optimizer_builder,
        train_dataloader_builder: DLBuilder = partial(
            default_dataloader_builder, shuffle=True
        ),
        valid_dataloader_builder: DLBuilder = default_dataloader_builder,
        device: Optional[str] = None,
    ):
        self.decode = decode
        self.collate = collate
        self.epochs = max_epochs
        self.train_dataloader_builder = train_dataloader_builder
        self.valid_dataloader_builder = valid_dataloader_builder

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        # loss_fn may register buffers (e.g. DetectionLoss.priors) that need
        # to live on the same device as the model/data; plain callables
        # have no `.to()` to call.
        self.loss_fn = (
            loss_fn.to(self.device)
            if isinstance(loss_fn, torch.nn.Module)
            else loss_fn
        )
        self.optimizer = optimizer_builder(self.model.parameters(), lr)

    def _loss(self, images: torch.Tensor, batch) -> torch.Tensor:
        preds = self.model(images.to(self.device))
        loss = self.loss_fn(preds, batch.to(self.device))
        return loss["loss"] if isinstance(loss, dict) else loss

    def fit(self, dataset, val_dataset=None) -> "SimpleTrainer":
        loader = self.train_dataloader_builder(dataset, self.collate.collate)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss, n_batches = 0.0, 0
            for images, batch in tqdm.tqdm(loader, desc=f"Epoch {epoch}"):
                loss = self._loss(images, batch)
                self.optimizer.zero_grad()
                if torch.is_tensor(loss):
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

            if n_batches:
                print(f"Epoch {epoch}, train loss: {epoch_loss / n_batches}")

            if val_dataset is not None:
                self._validate(val_dataset)

        return self

    def _validate(self, dataset) -> None:
        loader = self.valid_dataloader_builder(dataset, self.collate.collate)
        self.model.eval()
        val_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for images, batch in tqdm.tqdm(loader, desc="Validation"):
                loss = self._loss(images, batch)
                if torch.is_tensor(loss):
                    val_loss += loss.item()
                    n_batches += 1
        if n_batches:
            print(f"Valid loss: {val_loss / n_batches}")

    def predict(self, dataset) -> list:
        loader = self.valid_dataloader_builder(dataset, self.collate.collate)
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, _ in tqdm.tqdm(loader, desc="Prediction"):
                preds = self.decode(self.model(images.to(self.device)))
                results.extend(self.collate.un_batch_nms(preds))
        return results

    def predict_single(self, blob: torch.Tensor) -> list:
        self.model.eval()
        with torch.no_grad():
            preds = self.decode(self.model(blob.to(self.device)))
        return self.collate.un_batch_nms(preds)


def simple_engine(
    max_epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 2,
    num_workers: int = 0,
    train_dataloader_builder: Optional[DLBuilder] = None,
    valid_dataloader_builder: Optional[DLBuilder] = None,
    **knobs,
) -> Callable[[Baked], SimpleTrainer]:
    """Baked -> SimpleTrainer builder for build_detector(engine=...).
    batch_size/num_workers cover the common case (same vocabulary as the
    other engines); pass a *_dataloader_builder to take over loader
    construction entirely -- it wins over the knobs."""

    def build(baked: Baked) -> SimpleTrainer:
        return SimpleTrainer(
            model=baked.model,
            loss_fn=baked.loss,
            decode=baked.loss.decode,
            collate=baked.collate,
            max_epochs=max_epochs,
            lr=lr,
            train_dataloader_builder=train_dataloader_builder
            or partial(
                default_dataloader_builder,
                shuffle=True,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            valid_dataloader_builder=valid_dataloader_builder
            or partial(
                default_dataloader_builder,
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            **knobs,
        )

    return build
