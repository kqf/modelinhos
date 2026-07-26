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

    Sample[Annotation]                boxes in pixels, labels as strings
      | LabelEncoder.transform        normalize boxes to [0, 1], labels to ints
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
    list[Sample[Annotation]]          back to pixels and strings

Everything above ``model`` is the ``Detector``'s job (pixel space meets
normalized space exactly once, in ``LabelEncoder``); everything between
``model`` and ``loss`` is the engine's job; the recipe guarantees the
pieces agree.

Usage
-----

.. code-block:: python

    from modelinhos.engine.simple import simple_engine
    from modelinhos.models.ssdlite import SSDLITE
    from modelinhos.preprocess.lables import LabelEncoder
    from modelinhos.zoo import build_ssd

    lencoder = LabelEncoder(resolution=(480, 640)).fit(samples)
    detector = build_ssd(
        (480, 640),
        lencoder=lencoder,
        arch=SSDLITE,
        engine=simple_engine(epochs=10),
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
        engine=skorch_engine(max_epochs=10, lr=1e-3),
    )

``tests/models/test_trains.py`` runs this exact pipeline for every
model family crossed with every engine -- it is the reference example
of how the library is meant to be used.
