import gc
from pathlib import Path

import mkdocs_gen_files
from magicclass import get_button, get_function_gui, logging
from magicgui import magicgui
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

    ui.ImageMenu.open_simulator()
    for action in ui.simulator.SimulatorTools:
        btn = ui.simulator.SimulatorTools.native.widgetForAction(action.native)
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

    ### inspect local CFT ###
    ui.spectra_inspector.show()
    ui.spectra_inspector.load_spline(0)
    ui.spectra_inspector.peak_viewer.show_what = "Local-CFT"
    ui.spectra_inspector.width = 500
    ui.spectra_inspector.height = 525
    QtW.QApplication.processEvents()
    ui.spectra_inspector.peak_viewer.canvas.ylim = (16, 33)
    ui.spectra_inspector.peak_viewer.canvas.xlim = (28, 52)
    QtW.QApplication.processEvents()
    _imsave(ui.spectra_inspector.native, "inspect_local_cft")
    ui.spectra_inspector.peak_viewer._upsample_and_update_image(45, 23)
    ui.spectra_inspector.peak_viewer.canvas.xlim = (41, 49)
    ui.spectra_inspector.peak_viewer.canvas.ylim = (18, 28)
    QtW.QApplication.processEvents()
    _imsave(ui.spectra_inspector.native, "inspect_local_cft_upsampled")
    ui.spectra_inspector.peak_viewer.show_what = "Global-CFT"  # will be used later
    ui.copy_spline(0)
    ui.spline_fitter.fit(1, [[1, 0.5], [-0.2, 0.8], [1, 0.3]])
    ui.measure_radius(1)
    ui.local_cft_analysis(1, interval=50)
    ui.spectra_inspector.load_spline(1)
    ui.spectra_inspector.peak_viewer.show_what = "Local-CFT"
    ui.spectra_inspector.width = 500
    ui.spectra_inspector.height = 525
    QtW.QApplication.processEvents()
    ui.spectra_inspector.peak_viewer._upsample_and_update_image(45, 23)
    ui.spectra_inspector.peak_viewer.canvas.xlim = (41, 49)
    ui.spectra_inspector.peak_viewer.canvas.ylim = (18, 28)
    QtW.QApplication.processEvents()
    _imsave(ui.spectra_inspector.native, "inspect_local_cft_bad")

    ### magicgui widgets ###
    for meth in [
        ui.load_project,
        # fit, CFT, etc.
        ui.fit_splines,
        ui.refine_splines,
        ui.measure_radius,
        ui.set_radius,
        ui.measure_local_radius,
        ui.local_cft_analysis,
        ui.global_cft_analysis,
        # spline
        ui.clip_spline,
        # spline -> molecules
        ui.map_monomers,
        ui.map_along_spline,
        ui.map_along_pf,
        # molecules
        ui.filter_molecules,
        ui.paint_molecules,
        ui.split_molecules,
        ui.concatenate_molecules,
        ui.merge_molecule_info,
        ui.copy_molecules_features,
        # others
        ui.SplinesMenu.Config.update_default_config,
        ui.OthersMenu.configure_cylindra,
    ]:
        get_button(meth).changed.emit()
        gui = get_function_gui(meth)
        _imsave(gui.native, meth.__name__)
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

    ### canvas with monomers ###
    ui.parent_viewer.dims.ndisplay = 3
    _viewer_screenshot(ui, "viewer_01_monomer_mapped", canvas_only=True)
    ui.parent_viewer.dims.ndisplay = 2

    ### magicgui widgets ###
    for meth in [
        ui.sta.average_all,
        ui.sta.calculate_fsc,
        ui.sta.align_averaged,
        ui.sta.classify_pca,
        # align
        ui.sta.align_all,
        ui.sta.align_all_viterbi,
        ui.sta.align_all_annealing,
        # seam search
        ui.sta.seam_search,
        # landscape
        ui.sta.construct_landscape,
        ui.sta.run_align_on_landscape,
        ui.sta.run_viterbi_on_landscape,
        ui.sta.run_annealing_on_landscape,
        # simulator
        ui.simulator.create_empty_image,
        ui.simulator.generate_molecules,
    ]:
        get_button(meth).changed.emit()
        gui = get_function_gui(meth)
        _imsave(gui.native, meth.__name__)
        gui.close()

    ### Bigger widgets ###
    # Runner
    ui._runner.show()
    _imsave(ui._runner.native, "run_workflow_dialog")
    # Spline control
    _imsave(ui.SplineControl.native, "spline_control")
    # Local props
    _imsave(ui.LocalProperties.native, "local_props_gui")
    # Clipper
    ui.SplinesMenu.open_spline_clipper()
    _imsave(ui.spline_clipper.native, "spline_clipper")
    # Spectra inspector
    ui.AnalysisMenu.open_spectra_inspector()
    _imsave(ui.spectra_inspector.native, "spectra_inspector")
    # Image processor
    ui.image_processor.show()
    _imsave(ui.image_processor.native, "image_processor")

    # batch analyzer
    _imsave(ui.batch.constructor.native, "batch_constructor")

    # to avoid OpenGL rendering error
    for _ in range(10):
        QtW.QApplication.processEvents()
    gc.collect()


main()
