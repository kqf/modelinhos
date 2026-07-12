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


class SimpleTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        decode: Callable,
        collate,
        label_encoder,
        loss_fn: Callable = print,
        metrics: Optional[Callable] = None,
        optimizer_builder: Callable = default_optimizer_builder,
        lr: float = 1e-3,
        train_dataloader_builder: DLBuilder = default_dataloader_builder,
        valid_dataloader_builder: DLBuilder = default_dataloader_builder,
        epochs: int = 1,
        device: Optional[str] = None,
    ):
        self.decode = decode
        self.collate = collate
        self.lencoder = label_encoder
        self.loss_fn = loss_fn
        self.metrics_fn = metrics or MetricCollector
        self.train_dataloader_builder = train_dataloader_builder
        self.valid_dataloader_builder = valid_dataloader_builder
        self.epochs = epochs

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer_builder(self.model.parameters(), lr)

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
            validm = self.metrics_fn(self.lencoder.l2i)
            for images, targets in tqdm.tqdm(loader):
                images = images.to(self.device)
                preds = self.model(images)
                loss = self.loss_fn(targets, preds)

                self.optimizer.zero_grad()
                if torch.is_tensor(loss):
                    loss.backward()
                    self.optimizer.step()

                self._score(trainm, targets, preds)

            if val_dataset is not None:
                self._validate(validm, val_dataset)

            print(f"Epoch {epoch}, train mAP: {trainm.value().iloc[0]['mAP']}")
            print(f"Epoch {epoch}, train mAP: {validm.value().iloc[0]['mAP']}")

        return self

    def _validate(self, metrics_fn, dataset) -> None:
        loader = self.valid_dataloader_builder(dataset, self.collate.collate)
        self.model.eval()
        with torch.no_grad():
            for images, targets in tqdm.tqdm(loader):
                images = images.to(self.device)
                preds = self.model(images)
                self.loss_fn(targets, preds)
                self._score(metrics_fn, targets, preds)

    def predict(self, dataset) -> list:
        loader = self.valid_dataloader_builder(dataset, self.collate.collate)
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, _ in tqdm.tqdm(loader):
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


def build_trainer(
    loss_fn: Callable = print,
    metrics: Optional[Callable] = None,
    optimizer_builder: Callable = default_optimizer_builder,
    lr: float = 1e-3,
    train_dataloader_builder: DLBuilder = default_dataloader_builder,
    valid_dataloader_builder: DLBuilder = default_dataloader_builder,
    epochs: int = 1,
    device: Optional[str] = None,
) -> Callable[..., SimpleTrainer]:
    def _build(model, decode, collate, label_encoder) -> SimpleTrainer:
        return SimpleTrainer(
            model=model,
            decode=decode,
            collate=collate,
            label_encoder=label_encoder,
            loss_fn=loss_fn,
            metrics=metrics,
            optimizer_builder=optimizer_builder,
            lr=lr,
            train_dataloader_builder=train_dataloader_builder,
            valid_dataloader_builder=valid_dataloader_builder,
            epochs=epochs,
            device=device,
        )

    return _build
