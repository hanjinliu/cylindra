from cylindra import read_molecules, read_project, read_spline

from ._const import PROJECT_DIR_13PF


def test_read_molecules():
    read_molecules(PROJECT_DIR_13PF / "Mole-0.csv")


def test_read_spline():
    read_spline(PROJECT_DIR_13PF / "spline-0.json")


def test_read_project():
    read_project(PROJECT_DIR_13PF)
