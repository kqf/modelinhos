Detection models
================

This is a small model zoo of small detection models.

Design
------

The ``Detector`` is the public API: samples in, samples out, whatever
model family or training engine sits behind it. It is assembled from a
recipe, the definition of one model family whose pieces (anchors, loss,
heads, image encoder) are guaranteed to agree::

    DetectionRecipe.bake() -> Baked -> engine(Baked) -> Detector

In data flow terms, a sample goes through the label encoder, the model
and the loss's decoder, and comes back as a sample::

    Sample[Annotation]                relative xyxy boxes, string labels
      | LabelEncoder.transform
    Sample[TrainAnnotation]
      | SampleDataset + Collate
    (images, PerBatch)
      | model
    PerBatchEncoded
      | loss                          <- training stops here (backprop)
      | loss.decode
    PerBatch
      | Collate.un_batch_nms
    list[Sample[TrainAnnotation]]
      | LabelEncoder.inverse_transform
    list[Sample[Annotation]]

Usage
-----

Install:

.. code-block:: bash

    pip install modelinhos

and an example training:

.. code-block:: python

    from modelinhos.detector import build_detector
    from modelinhos.engine.simple import simple_engine
    from modelinhos.models.load import warm_start
    from modelinhos.models.ssdlite import SSDLITE
    from modelinhos.preprocess.labels import LabelEncoder

    lencoder = LabelEncoder().fit(samples)
    detector = build_detector(
        (480, 640),
        lencoder=lencoder,
        arch=SSDLITE,
        engine=simple_engine(max_epochs=10, batch_size=16),
        weights=warm_start(SSDLITE.pretraining, fresh=SSDLITE.diff),
    )
    detector.fit(samples)
    predictions = detector.transform(samples)
