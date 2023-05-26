import tempfile
from cylindra.widgets import CylindraMainWidget
from ._const import PROJECT_DIR_14PF


def test_project_export(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF, filter=False)
    with tempfile.TemporaryDirectory() as dirpath:
        ui.File.PEET.export_project(ui.parent_viewer.layers["Mono-0"], dirpath)
