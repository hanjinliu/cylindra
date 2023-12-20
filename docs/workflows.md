# Custom Workflows

Once you decided how to process your data, you may want to automate the workflow and
repetitively apply the same operations to many images. Of course, everything is Python,
so you can copy & paste the macro script to somewhere.

In `cylindra`, we provide a more elegant way to do this. You can define your own
workflows, browse and edit them in the GUI, and run them manually or programmatically.

## Define a Workflow

:material-arrow-right-thin-circle-outline: GUI: `Others > Workflows > Define workflow` or ++ctrl+k++ &rarr; ++ctrl+shift+d++

![Define workflow](images/define_workflow.png){ loading=lazy, width=400px }

In this dialog, you can define a workflow by writing a Python script. The script will be
saved in the user directory as a file of the given "file name". Since `cylindra` is
aware of all the files in the user directory, you don't have to keep them by yourself.

### A simple example

The `main` function defined in the workflow will be called when the workflow is run.
The first argument is always `ui`, the GUI object.

Following workflow is a simple example that measure the local properties of all the
splines and summarize the results in a histogram.

```python
from cylindra.widgets import CylindraMainWidget
import matplotlib.pyplot as plt

def main(ui: "CylindraMainWidget"):
    # fit and measure
    ui.fit_splines(splines="all")
    ui.measure_radius(splines="all", bin_size=2)
    ui.local_cft_analysis(splines="all", bin_size=2)

    # plot histograms of lattice spacings
    for spl in ui.splines:
        plt.hist(spl.props.get_loc("spacing"))
    plt.show()
```

This workflow can be called at any time &rarr; [Run Workflows](workflows.md#run-workflows).

### Workflow with input parameters

The main function accepts more input arguments. The function signature will be
automatically interpreted by [`magicgui`](https://pyapp-kit.github.io/magicgui/) and
converted into a dialog.

Following workflow is a completely redundant function as it does no more than the
[`open_image`][cylindra.widgets.main.CylindraMainWidget.open_image] method, but it
shows how to define a workflow with input parameters.

```python
from pathlib import Path
from cylindra.widgets import CylindraMainWidget

def main(
    ui: "CylindraMainWidget",
    path: Path,
    tilt_range: tuple[float, float] = (-60, 60),
):
    ui.open_image(path, tilt_range=tilt_range)

```

The resulting dialog will be like this.

![Workflow with args](images/workflow_with_args.png){ loading=lazy, width=400px }

The `path` argument, which is annotated with `Path` type, will be converted into a file
input box. The `tilt_range` argument, which is annotated with `tuple[float, float]`
type, will be converted into two float input boxes.

## View & Edit Workflows

:material-arrow-right-thin-circle-outline: GUI: `Others > Workflows > View/Edit workflows` or ++ctrl+k++ &rarr; ++ctrl+shift+e++

![Edit workflows](images/edit_workflow.png){ loading=lazy, width=400px }

Here you can see all the workflows defined in the user directory. You can also edit and
overwrite them here. The workflows will be updated immediately, without restarting the
application.

## Run Workflows

:material-arrow-right-thin-circle-outline: API: [`run_workflow`][cylindra.widgets.main.CylindraMainWidget.run_workflow]

:material-arrow-right-thin-circle-outline: GUI: `Others > Workflows > Run workflow` or ++ctrl+k++ &rarr; ++ctrl+shift+r++

Defined workflows show up in the `Others > Workflows` menu. You can also open the dialog
at `Others > Workflows > Run workflow` and select the workflow to run. If the workflow
does not have any input arguments, it will be run immediately. If it has, a new dialog
will be opened to ask for the input arguments.

!!! note
    The defined workflows can also be found in the command palette (++ctrl+p++) labeled
    as the file name.
