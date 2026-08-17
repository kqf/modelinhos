Detection models
================

This is a small model zoo of small detection models.

Design
------

Detection is inherently complex; the library hides that complexity
behind two public entities: ``DetectionRecipe`` (ties together the
pieces that must agree with each other) and ``Detector`` (dictates how
to infer on samples, so evaluation code never changes when the model
does). On the level of operations, a trainable detector is assembled
as::

    Recipe.bake() -> Baked -> engine(Baked) -> Detector(engine, iencoder, lencoder)

- ``DetectionRecipe`` -- the internally consistent definition of one
  model family: the anchors must be the ones the loss encodes against,
  which must match the head layout ``build_model`` produces, which must
  see pixels the way ``iencoder`` prepares them. Presets (``SSDLITE``,
  ``RETINANET``, ...) live next to their model definitions in
  ``modelinhos/models/``.
- ``Baked`` -- what ``bake(n_classes, resolution)`` produces once the
  label space and geometry are known: model, loss, collate, image
  encoder. Framework-agnostic, tensor-level. The loss owns the codec:
  decoding raw model output is ``loss.decode``.
- engine -- the owner of the optimization loop, satisfying the
  ``Engine`` protocol (``modelinhos/engine/``). Ships with a
  dependency-free ``SimpleTrainer`` plus optional skorch and lightning
  adapters; any other framework plugs in by writing one adapter module.
- ``Detector`` -- the stable public API: samples in, samples out,
  regardless of which model family or engine sits behind it.

On the level of data, one sample goes through::

    Sample[Annotation]                relative xyxy boxes, string labels
      | LabelEncoder.transform        labels to ints; boxes pass through
    Sample[TrainAnnotation]
      | SampleDataset[idx]            read pixels; iencoder: BGR uint8 HWC -> float CHW
    (image tensor, PerImage)
      | Collate.collate               stack images, pad annotations
    (images, PerBatch)                the training batch
      | model
    PerBatchEncoded                   raw per-anchor outputs (B, A, .)
      | loss                          <- training stops here (backprop)
      | loss.decode                   per-anchor outputs -> boxes/scores/labels
    PerBatch
      | Collate.un_batch_nms          strip padding, NMS
    list[Sample[TrainAnnotation]]
      | LabelEncoder.inverse_transform
    list[Sample[Annotation]]          back to string labels

Everything above ``model`` is the ``Detector``'s job (boxes stay
relative the whole way down and back -- only the labels change
representation, in ``LabelEncoder``); everything between ``model`` and
``loss`` is the engine's job; the recipe guarantees the pieces agree.

Usage
-----

Boxes are relative ([0, 1] xyxy) everywhere in the library and labels
are strings; ``LabelEncoder`` is purely the label <-> index mapping
(index 0 is reserved for background), so it needs no resolution:

.. code-block:: python

    from modelinhos.engine.simple import simple_engine
    from modelinhos.models.ssdlite import SSDLITE
    from modelinhos.preprocess.labels import LabelEncoder
    from modelinhos.zoo import build_ssd

    lencoder = LabelEncoder().fit(samples)
    detector = build_ssd(
        (480, 640),
        lencoder=lencoder,
        arch=SSDLITE,
        engine=simple_engine(max_epochs=10, batch_size=16, num_workers=8),
    )
    detector.fit(samples)
    predictions = detector.transform(samples)

Swap the training backend without touching anything else (skorch and
lightning are optional dependencies: ``pip install
modelinhos[skorch]`` / ``modelinhos[lightning]``):

.. code-block:: python

    from modelinhos.engine.skorch import skorch_engine

    detector = build_ssd(
        (480, 640),
        lencoder=lencoder,
        arch=SSDLITE,
        engine=skorch_engine(max_epochs=10, batch_size=16, num_workers=8),
    )

Every engine builder speaks the same common vocabulary (``max_epochs``,
``lr``, ``batch_size``, ``num_workers``); anything beyond that is passed
in the framework's own idiom (``iterator_train__*`` for skorch,
``trainer_kwargs`` for lightning, ``*_dataloader_builder`` callables for
full control over loader construction).

``tests/models/test_trains.py`` runs this exact pipeline for every
model family crossed with every engine -- it is the reference example
of how the library is meant to be used.

From start to production
------------------------

The scripts under ``trainings/`` and ``inference/`` walk the whole
loop; each one is flat and self-contained, and every training is tied
to a commit hash -- the script at that commit is the config.

1. **Data.** Datasets are json files of ``Sample`` objects (relative
   xyxy boxes, string labels) read with
   ``modelinhos.sample.read_samples``; ``modelinhos.coco.load_samples``
   downloads and converts a COCO split into that format on first use
   (``data/coco/annotations.json``).

2. **Study the data / recipe fit** before spending GPU time:
   ``trainings/study-ssd-misc.py`` wires ``modelinhos.analysis``
   (lint, box/label distributions, split divergence) and
   ``modelinhos.infos`` (model summary, matchability -- the recipe's
   own anchors and matcher run over every ground-truth box -- and the
   anchor advice derived from it) into one report, including the
   recall ceiling the anchor layout imposes.

3. **Train.** ``trainings/train-coco.py`` is the template: pick a
   recipe preset, ``warm_start(...)`` from a torchvision checkpoint or
   a previous run, add augmentations by ``replace()``-ing the recipe's
   ``augment``, and let the engine's own callbacks do the rest
   (validation mAP with precision/recall @0.5, OneCycleLR warmup,
   early stopping, checkpointing, MLflow logging).

4. **Evaluate and pick thresholds.**
   ``modelinhos.evaluation.mean_average_precision`` returns the full
   PR tables; ``trainings/evaluate-ssd-misc.py`` shows the flow --
   plot the PR curves, choose the operating threshold, then inspect
   per-image FP/FN at that threshold to find the failures. Every
   entry point takes a ``resolution=(h, w)`` and scales ground truth
   and predictions to it alike: the mAP backend computes VOC-style
   IoU with the inclusive ``+1`` pixel convention, so it needs a
   pixel space, and the numbers depend (weakly, ~1/object_size) on
   which one -- pin it to the deploy resolution.

5. **Export.** ``restore(checkpoint)`` is the strict loader for
   evaluation and export (any key or shape drift raises -- the
   checkpoint IS the model); ``inference/export.py`` exports every
   recipe to static-shape ONNX, benchmarked by the C++ runner in
   ``inference/infer.cpp``.
