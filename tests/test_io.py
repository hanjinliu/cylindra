from pathlib import Path
from cylindra import read_molecules, read_spline, read_project

TEST_DIR = Path(__file__).parent


def test_read_molecules():
    read_molecules(TEST_DIR / "test-project_results" / "Mono-0.csv")


def test_read_spline():
    read_spline(TEST_DIR / "test-project_results" / "spline-0.json")


def test_read_project():
    read_project(TEST_DIR / "test-project.json")
