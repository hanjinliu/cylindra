from cylindra.widgets import CylindraMainWidget
from cylindra_builtins import fetch


def test_fetch(ui: CylindraMainWidget):
    fetch.mt_13pf(ui, with_spline="roughly fitted")
    fetch.mt_13pf(ui, with_spline="fitted")
    fetch.mt_14pf(ui, with_spline="roughly fitted")
    fetch.mt_14pf(ui, with_spline="fitted")
