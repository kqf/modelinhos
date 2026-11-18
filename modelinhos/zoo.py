"""Label encoders over pretrained checkpoints' own categories. Paired
with build_detector(arch=<family recipe>, weights=warm_start(<checkpoint>))
they reproduce the pretrained detector exactly. Torchvision-native
reference predictions (the parity baseline) come from recipe.reference --
a plain predict function, not a Detector."""
