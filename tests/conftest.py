import os
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption("--show-viewer", default=False)


@pytest.fixture(scope="session", autouse=True)
def enable_console(request: pytest.FixtureRequest):
    # NOTE: magicclass currently check the IPython instance to determine the `show`
    # behavior.
    request.keywords["enable_console"] = True
    if sys.platform == "win32":
        # On Windows, we need to set the QT_QPA_PLATFORM to offscreen to avoid
        # OpenGL context issues in headless environments.
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
