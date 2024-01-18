from cylindra import collect_projects

from ._const import PROJECT_DIR_13PF, PROJECT_DIR_14PF


def test_io():
    seq = collect_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    assert len(seq) == 2
    assert seq[0].project_path == PROJECT_DIR_13PF
    assert seq[1].project_path == PROJECT_DIR_14PF


def test_concat():
    seq = collect_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    seq.collect_localprops(id="int")
    seq.collect_localprops(id="path", spline_details=True)
    seq.collect_globalprops(id="int")
    seq.collect_globalprops(id="path")
    seq.collect_globalprops(suffix="_g")
    seq.collect_joinedprops(spline_details=True)
    seq.collect_spline_coords(ders=(0, 1))

    collect_projects([PROJECT_DIR_13PF]) + collect_projects([PROJECT_DIR_14PF])


def test_mutable_sequence_methods():
    seq = collect_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    assert len(seq[0:2]) == 2
    p0 = seq[0]
    seq[0] = p0  # setitem
    del seq[0]  # delitem
    for _ in seq:
        pass
    seq.insert(0, p0)
    repr(seq)


def test_sequence_methods():
    seq = collect_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    mole = seq.collect_molecules(curvature=True)
    assert mole.count() > 500
    for (_, _), _ in seq.iter_molecules():
        pass


def test_molecules_items():
    seq = collect_projects([PROJECT_DIR_13PF, PROJECT_DIR_14PF])
    for item in seq.iter_molecules_with_splines(name_filter=lambda _: True):
        assert item.molecules.count() > 0
        assert item.spline is not None
        item.lattice_structure(("spacing",))
        item.local_vectors_longitudinal()
        item.local_vectors_lateral()
