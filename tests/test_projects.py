from cylindra import collect_projects
from ._const import PROJECT_DIR_13PF, PROJECT_DIR_14PF


def test_io():
    seq = collect_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    assert len(seq) == 2
    assert seq[0].project_path == PROJECT_DIR_13PF / "project.json"
    assert seq[1].project_path == PROJECT_DIR_14PF / "project.json"


def test_concat():
    seq = collect_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    seq.localprops()
    seq.globalprops()
    seq.all_props()
