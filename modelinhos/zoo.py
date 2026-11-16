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


def blaze_label_encoder(weights) -> LabelEncoder:
    """Label encoder over a blaze checkpoint's categories. Unlike the
    COCO checkpoints, blaze meta carries no background slot (the vanilla
    net emits a single sigmoid face channel), so indices start at 1 and
    0 keeps the background convention -- n_classes comes out as 2."""
    labels = weights.meta["categories"]
    return LabelEncoder(
        l2i={
            "__background__": 0,
            **{label: i for i, label in enumerate(labels, start=1)},
        },
    )
