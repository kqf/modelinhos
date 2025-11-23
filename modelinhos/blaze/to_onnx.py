import torch
import torch.onnx

from modelinhos.blaze.blazenet import BlazeNet


def export(
    height=128,
    width=128,
    back_model=False,
    opset_version=11,
    stem="blazenet",
):
    onnx_path = f"{stem}--{height}x{width}.onnx"

    # Create model instance
    model = BlazeNet(back_model=back_model)
    model.eval()

    # Dummy input with the requested resolution
    dummy = torch.randn(1, 3, height, width)

    print(f"Exporting BlazeNet with static input size: {height}x{width}")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["reg", "cls"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    print(f"Export complete -> {onnx_path}")


def main():
    resolutions = [
        (128, 128),
        (256, 256),
        (512, 512),
        (480, 480),
        (480, 640),
        (1024, 1024),
        # (1080, 1920), ~ This won't work
    ]
    for h, w in resolutions:
        export()


if __name__ == "__main__":
    main()
