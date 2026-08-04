import pytest
import torch

from modelinhos.models.load import restore, warm_start


class Tiny(torch.nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.head = torch.nn.Linear(4, n_classes)


@pytest.fixture
def checkpoint(tmp_path):
    # What engine checkpoints hold: the DetectionModel wrapper's state
    # dict, every key prefixed with "model."
    torch.save(
        {f"model.{k}": v for k, v in Tiny().state_dict().items()},
        tmp_path / "params.pt",
    )
    return tmp_path / "params.pt"


def test_restores_strictly_across_the_wrapper_prefix(checkpoint):
    model = restore(checkpoint)(Tiny())
    saved = torch.load(checkpoint, weights_only=True)
    for name, param in model.state_dict().items():
        assert torch.equal(param, saved[f"model.{name}"])


def test_restore_refuses_a_resized_head(checkpoint):
    with pytest.raises(RuntimeError):
        restore(checkpoint)(Tiny(n_classes=5))


def test_warm_start_fills_a_resized_head(checkpoint):
    model = warm_start(checkpoint)(Tiny(n_classes=5))
    saved = torch.load(checkpoint, weights_only=True)
    # load_with_mismatch grows rows by repeat_interleave, so the first
    # row of the resized head is still the checkpoint's first row
    assert model.head.weight.shape == (5, 4)
    assert torch.equal(
        model.head.weight[0],
        saved["model.head.weight"][0],
    )


def test_warm_start_accepts_torchvision_style_sources():
    class Enumish:
        def get_state_dict(self, progress):
            return Tiny(n_classes=5).state_dict()

    model = warm_start(Enumish())(Tiny(n_classes=2))
    assert model.head.weight.shape == (2, 4)
