from pathlib import Path

IMAGE_DIR = Path(__file__).parent.parent / "images"
PATH_13_3 = Path(__file__).parent.parent.parent / "tests" / "13pf_MT.tif"


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
    ui.GeneralInfo.value = ""  # hide
    ui.parent_viewer.window.resize(1200, 600)
    ui.parent_viewer.screenshot(
        IMAGE_DIR / "viewer_00_open_image.png", canvas_only=False
    )

    #
    ui.register_path(coords=[[17.974, 192.641, 24.25], [17.974, 34.149, 83.799]])
    ui.SplinesMenu.Fitting.fit_splines_manually()
    imsave(IMAGE_DIR / "fit_splines_manually.png", ui.spline_fitter.render())
    with suppress(RuntimeError):
        ui.parent_viewer.close()
        ui.close()


# To avoid running the script many times, only run if no images exist
FORCE = False
if next(IMAGE_DIR.glob("*.png"), None) is None or FORCE:
    main()
