from pathlib import Path

import mkdocs_gen_files
from pathlib import Path
from magicgui import magicgui
from magicclass import get_function_gui, get_button, logging
from qtpy import QtWidgets as QtW
from cylindra import instance

DOCS = Path(__file__).parent.parent
PATH_13_3 = DOCS.parent / "tests" / "13pf_MT.tif"
PATH_TEMP = DOCS.parent / "tests" / "beta-tubulin.tif"

assert PATH_13_3.exists()

_Logger = logging.getLogger("cylindra")


def _imsave(widget: QtW.QWidget, name: str):
    with mkdocs_gen_files.FilesEditor.current().open(f"images/{name}.png", "wb") as f:
        widget.grab().save(f.name, "png")


def _viewer_screenshot(ui, name: str, canvas_only: bool = True):
    with mkdocs_gen_files.FilesEditor.current().open(f"images/{name}.png", "wb") as f:
        ui.parent_viewer.screenshot(f.name, canvas_only=canvas_only)


def main():
    # `instance` will get the existing GUI created in `_dynamic_doc.py`
    ui = instance()
    if ui is None:
        return

    ui.clear_all()
    _Logger.widget.clear()

    # Image loader
    _imsave(ui.FileMenu.open_image_loader().native, "open_image_dialog")

    # Tool buttons
    for action in ui.Toolbar:
        btn = ui.Toolbar.native.widgetForAction(action.native)
        if not isinstance(btn, QtW.QToolButton) or len(action.name) == 0:
            continue
        _imsave(btn, f"toolbutton_{action.name}")

    # Open image -> screenshot viewer state
    ui.GeneralInfo.image_info.value = ""  # hide
    ui.parent_viewer.window.resize(1200, 600)
    QtW.QApplication.processEvents()
    _viewer_screenshot(ui, "viewer_00_open_image", canvas_only=False)

    # Fit splines manually window
    ui.register_path(coords=[[18.0, 192.6, 24.3], [18.0, 34.2, 83.8]])
    ui.SplinesMenu.Fitting.fit_splines_manually()
    _imsave(ui.spline_fitter.native, "fit_splines_manually")

    for method in [
        "load_project",
        "fit_splines",
        "refine_splines",
        "measure_radius",
        "measure_local_radius",
        "local_cft_analysis",
        "global_cft_analysis",
        "map_monomers",
        "map_centers",
        "map_along_pf",
    ]:
        meth = getattr(ui, method)
        get_button(meth).changed.emit()
        gui = get_function_gui(meth)
        _imsave(gui.native, method)
        gui.close()

    for method in [
        "define_workflow",
        "edit_workflow",
    ]:
        meth = getattr(ui.OthersMenu.Workflows, method)
        get_button(meth).changed.emit()
        gui = get_function_gui(meth)
        _imsave(gui.native, method)
        gui.close()

    if (dock := ui.parent_viewer.window._dock_widgets.get("workflow_gui")) is None:

        @magicgui
        def _workflow_gui(path: Path, tilt_range: tuple[float, float] = (-60, 60)):
            pass

        dock = ui.parent_viewer.window.add_dock_widget(
            _workflow_gui, name="workflow_gui"
        )
    dock.setFloating(True)
    _imsave(dock.widget(), "workflow_with_args")

    ui.sta.show()
    _imsave(ui.sta.native, "sta_widget")
    ui.sta.close()

    ui._runner.run(interval=12, n_refine=0, map_monomers=True)
    ui.parent_viewer.dims.ndisplay = 3
    _viewer_screenshot(ui, "viewer_01_monomer_mapped", canvas_only=True)
    ui.parent_viewer.dims.ndisplay = 2

    for method in [
        "average_all",
        "calculate_fsc",
        "align_averaged",
        "classify_pca",
        "align_all",
        "align_all_viterbi",
        "align_all_annealing",
        "construct_landscape",
        "run_align_on_landscape",
        "run_viterbi_on_landscape",
        "run_annealing_on_landscape",
    ]:
        meth = getattr(ui.sta, method)
        get_button(meth).changed.emit()
        gui = get_function_gui(meth)
        _imsave(gui.native, method)
        gui.close()

    # Runner
    ui._runner.show()
    _imsave(ui._runner.native, "run_workflow_dialog")
    # Spline control
    _imsave(ui.SplineControl.native, "spline_control")
    # Local props
    _imsave(ui.LocalProperties.native, "local_props_gui")

    QtW.QApplication.processEvents()


main()
