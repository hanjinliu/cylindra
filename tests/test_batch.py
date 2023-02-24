from cylindra.widgets import CylindraMainWidget
import tempfile
from pathlib import Path

TEST_PATH = Path(__file__).parent

def test_project_io(ui: CylindraMainWidget):
    ui.batch.construct_loader(
        paths=[
            (TEST_PATH / "13pf_MT.tif", 
             [TEST_PATH / "test-project_results" / "Mono-0.csv",
              TEST_PATH / "test-project_results" / "Mono-1.csv",])], 
        predicate=None,
        name='Loader',
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        path = root / "test-batch.json"
        ui.batch.save_batch_project(path)
        assert len(ui.batch._loaders) == 1
        ui.batch.load_batch_project(path)
        assert len(ui.batch._loaders) == 1
