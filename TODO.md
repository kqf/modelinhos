Things to do:

- [x] See what happens when we remove anchors from retina-net -> man is not detected, ties are kinda detected
- [x] Add the ssd model
- [x] Simplify the anchor tests
- [x] Fix the load with mismatch for when new(n_classes) > old(n_classes)
- [x] Add the fit/transform/predict wrappers
- [x] Add the notion of samples instead of operating on dicts
- [x] Fix plotting with Samples
- [x] Add the toy-example
- [x] Adapt the mAP calculation
- [x] Add per-sample map calculation.
- [x] Download the COCO evaluation dataset
- [x] Add a tool to select the thresholds
    - [x] Add FP / FN calculation @ threshold for each image to detect the failures
    - [x] Plot PR curve -- looking at these plots select the threshold
    - [x] Add FP / FN calculation per image @ threshold -- already evaluation
- [x] Use the batched version by default
- [x] Fix the tests to compare torchvision lite version with original with some tolerance
- [x] Add a separate test for inference when needed
- [x] Add bbox, class, label to batches
- [x] Add batch / un_batch functions
- [x] Eliminate box encoding / decoding to a separate file
- [x] Move the file_names to batches <--- Doesn't belong to batches tbh
- [x] Move the redundant parameters to decode
- [x] Write down the desired flow dataloader -> collate -> encode -> decode -> un_batch -> to_sample
- [x] Think of the composable pipeline instead of classes ? or vice versa -> class defines how to transform itself
- [x] Remove weights being exposed to the outside world in Detector
- [x] Add the actuall loss
- [x] Make the whole thing train on fake data
- [x] Add generalized decode function
- [x] Move to_preds to a model wrapper
- [x] Add the BlazeNet inference (anchors parametrization)
- [x] Implement the BlazeNet tests
- [x] Make Detection behave like any classification model
- [x] Finish trainings/study-ssd-misc.py -- needs modelinhos.infos
      (rule: if the answer changes when you swap the model, it goes to
      infos; data-only stays in analysis)
    - [x] summarize(model, resolution, n_classes): wrap torchinfo + a
          FLOP counter; own code only for latency, measured on the
          current machine (never predicted for the target device)
    - [x] matchability(samples, recipe, resolution): fact -- run the
          recipe's own priors + match_boxes (loss/matching.py) over
          every GT box, per-box matched-anchor counts
    - [x] anchor_advice(matched, geometry): verdict -- recall ceilings
          per class / size bucket, the serviceable size range implied
          by the steps (~2x finest to ~4x coarsest at the model
          resolution), the share of data outside it (only resolution
          fixes that: steps come from the backbone, sizes from data),
          suggested per-level sizes from the in-range box scales
    - [x] class_feasibility: dropped -- counts x matched in the
          anchor_advice table already is the joint judgement; the task
          label-space coverage check is a one-line set difference in
          the script (absent classes have no row to appear in)
    - [x] materialize(train, augmentation, draws, seed): sample the
          augmentation into a virtual split (draws ~ epochs), rerun
          the same fact functions on it -- lives in
          modelinhos.augment.infos (augmentation-facing, not model)
    - [x] Sanitize step: dropped -- sanitizing is the user's
          responsibility, the library only lints; the script's step 2
          stays a pass-through until its owner writes the jsons
    - [x] Wire visualize_labels / visualize_bboxes into the script
          (train vs test, at the model resolution) -- the script is
          hardwired to COCO now: the evaluation set randomly split
          into train/test (no train download), TORCHVISION_SSDLITE
          at its native 320x320, the checkpoint's own label space
P0 -- the train -> evaluate -> deploy loop is broken in the middle:

- [x] Load the checkpoints: `weights` now takes a loader from
      modelinhos.models.load carrying the intent -- warm_start(source)
      is mismatch-tolerant and only for training warm starts,
      restore(source) is strict (any key/shape drift raises) for
      evaluation and export. Source is a torchvision-style enum or a
      checkpoint path; the DetectionModel `model.` prefix is stripped
      at read time, so engines keep saving what they save and nothing
      unwrapped is ever written. Enums stay raw where only `meta` is
      read (coco_label_encoder, recipe.reference).
- [x] Fix packaging: `[tool.setuptools] packages = ["modelinhos"]`
      ships only the top-level package -- engine/, models/, loss/, ...
      are missing from a `pip install`. Switch to the find directive.

P1 -- deployable and measurable. Check retina vs ssd backbone effects

- [x] Fix the convoluted loading in build_model
- [x] Perhaps add SSDCUSTOM recipe that has exactly the same anchro structure as RETINANET (uses retina_anchors)
      - [x] Verify correctness
      - [x] Clean the code -> simplify
- [x] Download the coco dataset but unpack it to data/ location
- [x] DO the first trainings on COCO dataset:
  - [x] Train the COCO dataset with warmups probably on grayscale no normalization:
  - [x] Validatoin map
  - [x] Extend add the precision recall @p=0.5,
  - [x] LR scheduler
  - [x] Add augmentaitons
  - [x] Early stopping
- [x] Export to ONNX uniformly for all the recipes (only blaze/to_onnx
      exists today; depends on checkpoint loading)

P2 -- hygiene before there are external users (renames get expensive later):

- [x] Move modelinhos/train.py out of the library -- it is a script
      (cv2.imshow, joblib cache, __main__) predating trainings/, and it
      drags the dadinhos import into the package
- [x] Rename preprocess/lables.py -> labels.py (public import path with
      a typo, already spread through README and every script)
- [x] Update the readme: How to from start to production. The Usage
      example is already stale: LabelEncoder takes no `resolution`
      kwarg, and boxes are relative everywhere -- fix while at it.
- [x] Sanitize the BlazeNet: blaze/ is the vanilla reference only --
      to_onnx.py superseded by inference/export.py, the infer.py demo
      script folded into tests/models/test_blazenet.py, dead loaders
      dropped, predict_on_batch fixed to return every image's faces
- [x] Make the BlazeNet a part of a group: BLAZEFACE recipe (vanilla
      front model, MediaPipe box codec) with the original blaze/
      inference path demoted to recipe.reference for parity tests, plus
      a RETINANET blaze flavor (BlazePure: stride-32 stage + generic
      heads on retina_anchors, no reference -- no upstream exists) and
      build_blaze/blaze_label_encoder in the zoo
- [ ] Make the pipeline generic

Someday:

- [ ] Switch the mAP backend to geometric IoU (torchmetrics'
      MeanAveragePrecision or an in-house AP over torchvision box_iou):
      the mean_average_precision package uses the VOC inclusive-pixel
      (+1) IoU convention, which is why evaluation currently needs a
      `resolution` parameter and inflates IoU for small objects (~1/size
      per dimension). After the switch, evaluate directly on relative
      coordinates and drop `resolution` except for pixel-denominated
      breakdowns (area buckets, sub-Npx diagnostics). Expensive --
      until then, keep the VOC-style (+1) backend and its `resolution`.

- [x] Add the fcos inference
- [x] Drop the zoo builders entirely: give DetectionRecipe a
      `pretrained` field (the checkpoint enum it mirrors) and default
      build_detector's weights to warm_start(arch.pretrained), so
      build_ssd/build_retina/build_fcos/build_blaze (now just partials
      pinning arch + weights) become plain build_detector(arch=...).
      Trade-off: couples recipes to checkpoints -- the trimmed flavors
      (SSDLITE, SSDLARGE, RETINANET, RETINANET_F/B, FCOS_DELTA) currently
      borrow their family's checkpoint via the zoo defaults, so each
      would need `pretrained` set explicitly; a `None` would mean
      from_scratch.

Decided against (deliberate, don't re-propose):

- No detector cards / config persistence next to weights: every
  training is tied to a commit hash, the script at that commit IS the
  config.

Review findings:

R0 -- breaks in real use (fit -> predict -> checkpoint path first):

- [ ] Detector.transform cannot predict on unlabelled images: it
      routes through the training SampleDataset, whose field_specs
      raises "no annotated sample" (data.py:39). An unseen label is a
      bare KeyError, and predictions come back with
      file_name="fake-file.png" (containers.py:121) -- reattach the
      input file names.
- [ ] `resolution` is not enforced: it sizes the anchors, but the
      default image encoder never resizes (preprocess/image.py:17). An
      image of another size fails at decode with a shape mismatch,
      mixed sizes fail in torch.stack. Only grayscale_image_encoder
      resizes.
- [ ] num_workers > 0 fails under spawn/forkserver (macOS and Windows
      always, Linux from Python 3.14): the image encoder is built from
      lambdas and cannot be pickled; the augment closures have the same
      shape. The README advertises num_workers=8.
- [ ] RetinaNetPure always downloads ImageNet ResNet50 weights
      (models/retinanet.py:71): from_scratch is not from scratch, and
      restore / export need network. Contradicts the recipe contract
      (detector.py:107); affects the RetinaNet and FCOS flavors.
- [ ] The SSD matcher gives background-only images zero negatives, so
      zero loss (loss/matching.py:192) -- the field_specs effort to
      support empty images buys nothing for the mined losses.
- [ ] The skorch engine cannot predict before fit: a pretrained zoo
      detector raises NotInitializedError.

R1 -- architectural flaws:

- [ ] The loss owns decode and the score threshold: `th` is baked in
      at build time (detector.py:131), so changing the operating
      threshold means rebuilding the detector, while the README
      workflow is "evaluate, then pick thresholds". The NMS IoU of 0.5
      is hard-coded in Collate() and unreachable from the recipe or
      build_detector.
- [ ] Per-field decode is strained: softmax is computed twice, the
      threshold is encoded as label = -1, and FCOS smuggles centerness
      through by concatenating it onto the class logits, which to_preds
      then duplicates into two slots.
- [ ] detector.py is a hub (DetectionModel, Detector, Baked,
      DetectionRecipe, build_detector, torchvision_reference); every
      model and engine imports it. Give Baked and the recipe their own
      module.
- [ ] The Engine protocol is too wide and too narrow: each engine
      re-implements the same predict loop (could live once in
      Detector), while device is not in the protocol (save already is).
- [ ] evaluation.py mixes metrics with blocking plt.show() views, and
      async_mode=True starts a manager and worker process on every
      mean_average_precision() call (once per validation epoch in the
      training scripts).

R2 -- misconfigurations:

- [ ] pyproject dependencies: pandas and typing_extensions are
      imported but undeclared (transitive only); dadinhos, joblib and
      onnx are declared but unused by the package; mlflow is in every
      extra but never imported by the library; no requires-python
      (the code needs >= 3.10).
- [ ] requirements.txt: pytest-coverage is a placeholder alias for
      pytest-cov.
- [ ] CI (tests.yml): tests only 3.12 while the tooling targets 3.10;
      tests run against the source tree, not the installed wheel (how
      the packages= bug slipped through); every run re-downloads
      pretrained weights with no cache; checkout@v3 and
      setup-python@v4 are outdated.
- [ ] publish.yml fires on `release: created` with no test gate.
- [ ] Linters: pyright has no config, so locally it scans cachedir/
      (9 errors); mypy runs with ignore_missing_imports in an isolated
      env, so torch is Any and it checks very little; ruff ignores
      B008 / RUF009 but neither rule set is selected; pre-commit-hooks
      is pinned to a 2021 release.

R3 -- loose ends:

- [ ] Dead code: masked_loss (also swaps pred and target,
      loss/subloss.py:45), retina_confidence_loss (divides by zero
      with no positives), Collate.un_batch, AnchorConfig (test only),
      the empty tests/test_all.py, and DoNothingEncoder -- the
      Detector default, which cannot work with SampleDataset.
- [ ] Stale comments: the 30-line BatchNorm note about a test in
      models/ssdlite.py:272, the dangling `reference` field comment at
      detector.py:122, the mypy v0.921 mention in
      augment/__init__.py:23, and the torchvision adapter docstring
      pointing at a nonexistent albumentations_augment.
- [ ] Typos in public names (renames get expensive later): overalp,
      iterative_mathing, bulid_retinanet, tvison_anchors. RETINANET_F/B
      in blazenet also collide with retinanet.RETINANET -- export.py
      has to alias them.
- [ ] Untested: both augment adapters, coco.py, and inference on
      unlabelled data.
- [ ] README drift: it says test_trains.py crosses every family with
      every engine; it runs only simple_engine, the engines are
      crossed for one model in test_engine.py.
- [ ] trainings/ drift: the scripts call net.predict(X), which now
      raises TypeError -- the method is predict_samples.
