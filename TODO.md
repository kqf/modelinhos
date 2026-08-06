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
- [ ] Switch the mAP backend to geometric IoU (torchmetrics'
      MeanAveragePrecision or an in-house AP over torchvision box_iou):
      the mean_average_precision package uses the VOC inclusive-pixel
      (+1) IoU convention, which is why evaluation currently needs a
      `resolution` parameter and inflates IoU for small objects (~1/size
      per dimension). After the switch, evaluate directly on relative
      coordinates and drop `resolution` except for pixel-denominated
      breakdowns (area buckets, sub-Npx diagnostics).
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

- [ ] Fix the convoluted loading in build_model
- [x] Perhaps add SSDCUSTOM recipe that has exactly the same anchro structure as RETINANET (uses retina_anchors)
      - [ ] Verify correctness
      - [ ] Clean the code -> simplify
- [ ] Download the coco dataset but unpack it to data/ location
- [ ] DO the first trainings on COCO dataset:
  - [ ] Train the COCO dataset with warmups probably on grayscale no normalization:
  - [ ] Validatoin map
  - [ ] Extend add the precision recall @p=0.5,
  - [ ] LR scheduler
  - [ ] Add augmentaitons
  - [ ] Early stopping
- [ ] Export to ONNX uniformly for all the recipes (only blaze/to_onnx
      exists today; depends on checkpoint loading)

P2 -- hygiene before there are external users (renames get expensive later):

- [x] Move modelinhos/train.py out of the library -- it is a script
      (cv2.imshow, joblib cache, __main__) predating trainings/, and it
      drags the dadinhos import into the package
- [ ] Rename preprocess/lables.py -> labels.py (public import path with
      a typo, already spread through README and every script)
- [ ] Update the readme: How to from start to production. The Usage
      example is already stale: LabelEncoder takes no `resolution`
      kwarg, and boxes are relative everywhere -- fix while at it.
- [ ] Sanitize the BlazeNet
- [ ] Make the BlazeNet a part of a group: models/blazenet.py has the
      weights enum but no DetectionRecipe; blaze/ still carries its own
      infer/postprocessing path outside the Recipe/Detector flow
- [ ] Make the pipeline generic

Someday:

- [ ] Add the fcos inference

Decided against (deliberate, don't re-propose):

- No detector cards / config persistence next to weights: every
  training is tied to a commit hash, the script at that commit IS the
  config.
- No geometric-IoU mAP switch: too expensive, keeping the
  mean_average_precision VOC-style (+1) backend and its `resolution`
  parameter.
