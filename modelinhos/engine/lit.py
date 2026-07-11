"""Lightning adapter. The LightningModule is a pure (model, loss)
shell -- loaders and pl.Trainer live in the engine, and decode/NMS stay
outside the loop entirely (evaluate after fit via Detector.transform).
Importing this module requires lightning
(pip install modelinhos[lightning])."""

from functools import partial
from typing import Callable, Optional

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # older installs use the standalone package name
    try:
        import pytorch_lightning as pl
    except ImportError as e:
        raise ImportError(
            "the lightning engine needs lightning installed -- "
            "pip install modelinhos[lightning]"
        ) from e

from modelinhos.detector import Baked
from modelinhos.engine.simple import (
    DLBuilder,
    default_dataloader_builder,
    default_optimizer_builder,
)


class LitDetection(pl.LightningModule):
    """Holds only what Lightning needs to optimize: the model, the loss
    and the optimizer recipe. Assigning the loss as a submodule moves
    its buffers (DetectionLoss.priors) along with the model."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Callable,
        lr: float = 1e-3,
        optimizer_builder: Callable = default_optimizer_builder,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.optimizer_builder = optimizer_builder

    def forward(self, images):
        return self.model(images)

    def _shared_step(self, batch, stage: str):
        images, targets = batch
        loss = self.loss(self.model(images), targets)
        loss = loss["loss"] if isinstance(loss, dict) else loss
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=(stage == "train"),
            batch_size=images.shape[0],
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return self.optimizer_builder(self.parameters(), self.lr)


class LightningEngine:
    """Maps the Engine protocol onto pl.Trainer. Prediction is a plain
    eval loop -- no Trainer needed to run a forward pass."""

    def __init__(
        self,
        module: LitDetection,
        decode: Callable,
        collate,
        train_dataloader_builder: DLBuilder,
        valid_dataloader_builder: DLBuilder,
        trainer_kwargs: dict,
    ):
        self.module = module
        self.decode = decode
        self.collate = collate
        self.train_dataloader_builder = train_dataloader_builder
        self.valid_dataloader_builder = valid_dataloader_builder
        self.trainer_kwargs = trainer_kwargs

    def fit(self, dataset, val_dataset=None) -> "LightningEngine":
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
            self.module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        return self

    def predict(self, dataset) -> list:
        loader = self.valid_dataloader_builder(dataset, self.collate.collate)
        self.module.eval()
        results = []
        with torch.no_grad():
            for images, _ in loader:
                preds = self.decode(self.module(images.to(self.module.device)))
                results.extend(self.collate.un_batch_nms(preds))
        return results

    def predict_single(self, blob: torch.Tensor) -> list:
        self.module.eval()
        with torch.no_grad():
            preds = self.decode(self.module(blob.to(self.module.device)))
        return self.collate.un_batch_nms(preds)


def lightning_engine(
    max_epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 2,
    optimizer_builder: Callable = default_optimizer_builder,
    trainer_kwargs: Optional[dict] = None,
) -> Callable[[Baked], LightningEngine]:
    """Baked -> LightningEngine builder for build_detector(engine=...).
    trainer_kwargs are merged over the quiet defaults (no logger, no
    checkpointing)."""

    def build(baked: Baked) -> LightningEngine:
        return LightningEngine(
            module=LitDetection(
                model=baked.model,
                loss=baked.loss,
                lr=lr,
                optimizer_builder=optimizer_builder,
            ),
            decode=baked.loss.decode,
            collate=baked.collate,
            train_dataloader_builder=partial(
                default_dataloader_builder,
                shuffle=True,
                batch_size=batch_size,
            ),
            valid_dataloader_builder=partial(
                default_dataloader_builder,
                batch_size=batch_size,
            ),
            trainer_kwargs={
                "max_epochs": max_epochs,
                "logger": False,
                "enable_checkpointing": False,
                **(trainer_kwargs or {}),
            },
        )

    return build
