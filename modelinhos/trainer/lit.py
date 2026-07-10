from typing import Callable, Optional

import torch

from modelinhos.preprocess.lables import SampleEncoder

try:
    import lightning.pytorch as pl
except ImportError:  # older installs use the standalone package name
    import pytorch_lightning as pl

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
    params, lr: float = 1e-3
) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=lr)


class LitDetectionModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        decode: Callable,
        collate,
        label_encoder: SampleEncoder,
        loss_fn: Callable = print,
        metrics: Optional[Callable] = None,
        optimizer_builder: Callable = default_optimizer_builder,
        lr: float = 1e-3,
        train_dataloader_builder: DLBuilder = default_dataloader_builder,
        valid_dataloader_builder: DLBuilder = default_dataloader_builder,
        trainer_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.decode = decode
        self.collate = collate
        self.label_encoder = label_encoder
        self.metrics_fn = metrics
        self.optimizer_builder = optimizer_builder
        self.lr = lr
        self.train_dataloader_builder = train_dataloader_builder
        self.valid_dataloader_builder = valid_dataloader_builder
        self.trainer_kwargs = trainer_kwargs or {}

    def forward(self, images):
        return self.model(images)

    def _shared_step(self, batch, stage: str):
        images, targets = batch
        preds = self.model(images)
        loss = self.loss_fn(targets, preds)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=(stage == "train"),
        )

        if self.metrics_fn is not None:
            true = self.label_encoder.inverse_transform(
                self.collate.un_batch(targets),
            )
            pred = self.label_encoder.inverse_transform(
                self.collate.un_batch_nms(self.decode(preds)),
            )
            self.metrics_fn(true, pred)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        images, _ = batch
        preds = self.decode(self.model(images))
        return self.collate.un_batch_nms(preds)

    def configure_optimizers(self):
        return self.optimizer_builder(self.parameters(), self.lr)

    def fit(self, dataset, val_dataset=None) -> "LitDetectionModule":
        train_loader = self.train_dataloader_builder(
            dataset, self.collate.collate
        )
        val_loader = (
            self.valid_dataloader_builder(val_dataset, self.collate.collate)
            if val_dataset is not None
            else None
        )
        runner = pl.Trainer(**self.trainer_kwargs)
        runner.fit(
            self, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        return self

    def predict(self, dataset) -> list:
        loader = self.valid_dataloader_builder(dataset, self.collate.collate)
        runner = pl.Trainer(
            **{
                **self.trainer_kwargs,
                "logger": False,
                "enable_checkpointing": False,
            }
        )
        batches = runner.predict(self, dataloaders=loader)
        return [sample for batch in batches for sample in batch]

    def predict_single(self, blob: torch.Tensor) -> list:
        self.eval()
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
    trainer_kwargs: Optional[dict] = None,
) -> Callable[..., LitDetectionModule]:
    def _build(model, decode, collate, label_encoder) -> LitDetectionModule:
        return LitDetectionModule(
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
            trainer_kwargs=trainer_kwargs,
        )

    return _build
