from pathlib import Path
from cylindra import read_molecules, read_spline, read_project

TEST_DIR = Path(__file__).parent
PROJECT_DIR = TEST_DIR / "test_project"


def test_read_molecules():
    read_molecules(PROJECT_DIR / "Mono-0.csv")


def test_read_spline():
    read_spline(PROJECT_DIR / "spline-0.json")


def test_read_project():
    read_project(PROJECT_DIR)
