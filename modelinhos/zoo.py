"""Label encoders over pretrained checkpoints' own categories. Paired
with build_detector(arch=<family recipe>, weights=warm_start(<checkpoint>))
they reproduce the pretrained detector exactly. Torchvision-native
reference predictions (the parity baseline) come from recipe.reference --
a plain predict function, not a Detector."""

from modelinhos.preprocess.labels import LabelEncoder


def coco_label_encoder(weights) -> LabelEncoder:
    """Label encoder over the checkpoint's own COCO categories -- 91
    channels, so the mismatch-tolerant load is a passthrough."""
    labels = weights.meta["categories"]
    return LabelEncoder(
        l2i={label: i for i, label in enumerate(labels)},
    )
