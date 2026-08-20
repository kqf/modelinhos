import os

import pytest


@pytest.fixture
def headless():
    return os.environ.get("DISPLAY", "") == ""
