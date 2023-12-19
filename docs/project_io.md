# Load & Save Projects

In `cylindra`, a project is managed by a `CylindraProject` instance.

## Save the Session as a Project

:material-arrow-right-thin-circle-outline: API: [`save_project`][cylindra.widgets.main.CylindraMainWidget.save_project]

:material-arrow-right-thin-circle-outline: GUI: `File > Save Project` or ++ctrl+k++ &rarr; ++ctrl+s++

If you want to save the current session as a project, you can use this method to save
the current state.

- `project.json` ... The main project file, which has the description of this project.
- `default_spline_config.json` ... The default `SplineConfig`. See [here](configure.md)
  for the details.
- `globalprops.csv` ... The global properties of the splines.
- `localprops.csv` ... The local properties of the splines.
- CSV or parquet files ... Coordinates and features of molecules.
- `script.py` ... An executable Python script to reproduce the session.
- `spline-*.json` ... The spline objects in JSON format (not human readable, should
  always be read by [from_json][cylindra.components.BaseComponent.from_json]) method.

## Load Project

:material-arrow-right-thin-circle-outline: API: [`load_project`][cylindra.widgets.main.CylindraMainWidget.load_project]

:material-arrow-right-thin-circle-outline: GUI: `File > Load Project` or ++ctrl+k++ &rarr; ++ctrl+p++

## Stash the Session

:material-arrow-right-thin-circle-outline: GUI: `File > Stash`

You may want to temporarily save the session for later use. In `cylindra`, some "stash"
operations are available. Stashed sessions are cached in the user directory, and can be
readily loaded from the GUI.

!!! note
    If you already know the `git stash` command, you should be familiar with this.
