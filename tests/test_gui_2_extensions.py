from contextlib import suppress
import tempfile
from pathlib import Path
from cylindra.ext import CommandNotFound
from cylindra.widgets import CylindraMainWidget
from ._const import PROJECT_DIR_14PF, TEST_DIR


def test_PEET(ui: CylindraMainWidget):
    ui.load_project(PROJECT_DIR_14PF)
    peet = ui.FileMenu.PEET
    with tempfile.TemporaryDirectory() as tmpdir:
        with suppress(CommandNotFound):
            peet.save_monomers(Path(tmpdir), ui.mole_layers.nth(0))
        with suppress(CommandNotFound):
            peet.save_all_monomers(Path(tmpdir))
        with suppress(CommandNotFound):
            peet.export_project(
                ui.mole_layers.nth(0),
                Path(tmpdir),
                template_path=TEST_DIR / "beta-tubulin.mrc",
                mask_params=(0.3, 0.8),
            )
