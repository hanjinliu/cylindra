from cylindra import collect_projects
from ._const import PROJECT_DIR_13PF, PROJECT_DIR_14PF


def test_io():
    seq = collect_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    assert len(seq) == 2
    assert seq[0].project_path == PROJECT_DIR_13PF / "project.json"
    assert seq[1].project_path == PROJECT_DIR_14PF / "project.json"


def test_concat():
    seq = collect_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    seq.collect_localprops(id="int")
    seq.collect_localprops(id="path")
    seq.collect_globalprops(id="int")
    seq.collect_globalprops(id="path")
    seq.collect_globalprops(suffix="_g")
    seq.collect_joinedprops()
    seq.collect_spline_coords(ders=(0, 1))


def test_mutable_sequence_methods():
    seq = collect_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    assert len(seq[0:2]) == 2
    p0 = seq[0]
    seq[0] = p0  # setitem
    del seq[0]  # delitem
    for proj in seq:
        pass
    seq.insert(0, p0)
    repr(seq)


def test_sequence_methods():
    seq = collect_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    seq.sta_loader()
    for (i, s), mole in seq.iter_molecules(spline_props=["spacing"]):
        pass
