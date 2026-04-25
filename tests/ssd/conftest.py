import pytest


@pytest.fixture
def resolution() -> tuple[int, int]:
    # This is magic resolution to avoid additional geometric conversion
    return 800, 1088
    return 320, 320
