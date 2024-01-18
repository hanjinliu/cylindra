# Load & Save Projects

In `cylindra`, a project is managed by a `CylindraProject` instance.

## Save the Current Session as a Project

:material-arrow-right-thin-circle-outline: API: [`save_project`][cylindra.widgets.main.CylindraMainWidget.save_project]

:material-arrow-right-thin-circle-outline: GUI: `File > Save Project` or ++ctrl+k++ &rarr; ++ctrl+s++

If you want to save the current session as a project, you can use this method to save
the current state.

- `project.json` ... The main project file, which has the description of this project.
- `default_spline_config.json` ... The default `SplineConfig`. See [here](spline/config.md)
  for the details.
- `globalprops.csv` ... The global properties of the splines.
- `localprops.csv` ... The local properties of the splines.
- CSV or parquet files ... Coordinates and features of molecules.
- `script.py` ... An executable Python script to reproduce the session.
- `spline-*.json` ... The spline objects in JSON format (not human readable, should
  always be read by [from_json][cylindra.components.BaseComponent.from_json]) method.

The project can be a directory, a zip file or a tar file.

## Load Project

:material-arrow-right-thin-circle-outline: API: [`load_project`][cylindra.widgets.main.CylindraMainWidget.load_project]

:material-arrow-right-thin-circle-outline: GUI: `File > Load Project` or ++ctrl+k++ &rarr; ++ctrl+p++

Saved projects can be loaded to the GUI. Loading a project does not rerun the script.
It uses the saved files to reproduce the session.

![Load Project](images/load_project.png)


## Stash the Session

:material-arrow-right-thin-circle-outline: GUI: `File > Stash`

You may want to temporarily save the session for later use. In `cylindra`, some "stash"
operations are available. Stashed sessions are cached in the user directory, and can be
readily loaded from the GUI.

!!! note
    If you already know the `git stash` command, you should be familiar with this.

# Reuse the Existing Projects

:material-arrow-right-thin-circle-outline: API: [`load_project_for_reanalysis`][cylindra.widgets.main.CylindraMainWidget.load_project_for_reanalysis]

:material-arrow-right-thin-circle-outline: GUI: `Analysis > Load Project for Reanalysis`

Drawing splines is a most time-consuming step. `cylindra` is implemented with a method
that automatically filter the script in a project file and only run until it reaches
any non-manual line. For example, if you have a project file that contains a `script.py`
file with a `main` function like this:

``` python
def main(ui):
    ui.open_image(path='path/to/image.tif', ...)
    ui.register_path(coords=[[19, 190, 29], [19, 100, 50]], ...)
    ui.register_path(coords=[[20, 100, 32], [20, 190.0, 63]], ...)
    ui.fit_splines(splines='all', ...)
    ui.refine_splines(splines='all', ...)
```

then, the filtered script will be:

``` python
def main(ui):
    ui.open_image(path='path/to/image.tif', ...)
    ui.register_path(coords=[[19, 190, 29], [19, 100, 50]], ...)
    ui.register_path(coords=[[20, 100, 32], [20, 190.0, 63]], ...)
```
