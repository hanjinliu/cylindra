from pathlib import Path
from magicclass import get_function_gui, get_button

IMAGE_DIR = Path(__file__).parent.parent / "docs" / "images"
PATH_13_3 = Path(__file__).parent.parent / "tests" / "13pf_MT.tif"

assert IMAGE_DIR.exists()
assert PATH_13_3.exists()


def main():
    from cylindra import start
    from contextlib import suppress
    from imageio import imsave

    ui = start()
    # Image loader
    loader = ui.FileMenu.open_image_loader()
    imsave(IMAGE_DIR / "open_image_dialog.png", loader.render())

    # Open image -> screenshot viewer state
    ui.open_image(PATH_13_3)
    ui.GeneralInfo.image_info.value = ""  # hide
    ui.parent_viewer.window.resize(1200, 600)
    ui.parent_viewer.screenshot(
        IMAGE_DIR / "viewer_00_open_image.png", canvas_only=False
    )

    # Tool buttons
    ui.Toolbar
    # Fit splines manually window
    ui.register_path(coords=[[18.0, 192.6, 24.3], [18.0, 34.2, 83.8]])
    ui.SplinesMenu.Fitting.fit_splines_manually()
    imsave(IMAGE_DIR / "fit_splines_manually.png", ui.spline_fitter.render())

    for method in [
        "fit_splines",
        "refine_splines",
        "measure_radius",
        "measure_local_radius",
        "local_cft_analysis",
        "global_cft_analysis",
        "map_monomers",
    ]:
        meth = getattr(ui, method)
        get_button(meth).changed.emit()
        gui = get_function_gui(meth)
        imsave(IMAGE_DIR / f"{method}.png", gui.render())
        gui.close()

    for method in [
        "define_workflow",
        "edit_workflow",
    ]:
        meth = getattr(ui.OthersMenu.Workflows, method)
        get_button(meth).changed.emit()
        gui = get_function_gui(meth)
        imsave(IMAGE_DIR / f"{method}.png", gui.render())
        gui.close()

    ui.sta.show()
    imsave(IMAGE_DIR / "sta_widget.png", ui.sta.render())
    ui.sta.close()

    ui._runner.run(interval=12)

    # Runner
    ui._runner.show()
    imsave(IMAGE_DIR / "run_workflow_dialog.png", ui._runner.render())
    # Spline control
    imsave(IMAGE_DIR / "spline_control.png", ui.SplineControl.render())
    # Local props
    imsave(IMAGE_DIR / "local_props_gui.png", ui.LocalProperties.render())

    # Make sure windows are closed
    with suppress(RuntimeError):
        ui.parent_viewer.close()
        ui.close()


if __name__ == "__main__":
    main()
