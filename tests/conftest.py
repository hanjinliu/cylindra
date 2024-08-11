import pytest


def pytest_addoption(parser):
    parser.addoption("--show-viewer", default=False)


@pytest.fixture(scope="session", autouse=True)
def enable_console(request: pytest.FixtureRequest):
    # NOTE: magicclass currently check the IPython instance to determine the `show`
    # behavior.
    request.keywords["enable_console"] = True
