import os

import pytest


@pytest.fixture
def resolution() -> tuple[int, int]:
    # This is magic resolution to avoid additional geometric conversion
    return 800, 1088


@pytest.fixture
def headless():
    return os.environ.get("DISPLAY", "") == ""
