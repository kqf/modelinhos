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
- [ ] Add a tool to select the thresholds
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
- [ ] Make Detection behave like any classification model
- [ ] Add the fcos inference
- [ ] Sanitize the BlazeNet
- [ ] Make the BlazeNet a part of a group
- [ ] Finish trainings/study-ssd-misc.py -- needs modelinhos.inspect
      (rule: if the answer changes when you swap the model, it goes to
      inspect; data-only stays in analysis)
    - [ ] summarize(model, resolution, n_classes): wrap torchinfo + a
          FLOP counter; own code only for latency, measured on the
          current machine (never predicted for the target device)
    - [ ] matchability(samples, recipe, resolution): fact -- run the
          recipe's own priors + match_boxes (loss/matching.py) over
          every GT box, per-box matched-anchor counts
    - [ ] anchor_advice(matched, geometry): verdict -- recall ceilings
          per class / size bucket, the serviceable size range implied
          by the steps (~2x finest to ~4x coarsest at the model
          resolution), the share of data outside it (only resolution
          fixes that: steps come from the backbone, sizes from data),
          suggested per-level sizes from the in-range box scales
    - [ ] class_feasibility(counts, matched): verdict -- owns the task
          label space (l2i), checks train covers it, judges sample
          count x matchability jointly
    - [ ] materialize(train, augmentation, draws, seed): sample the
          augmentation into a virtual split (draws ~ epochs), rerun
          the same fact functions on it
    - [ ] Real sanitize step: drop sub-floor boxes, collapse labels to
          the single "object" class, write sanitized jsons -- the
          script's step 2 is currently a pass-through of linted.good
    - [ ] Wire visualize_labels / visualize_bboxes into the script
          (train vs test, at the model resolution)
