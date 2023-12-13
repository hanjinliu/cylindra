# Installation

### Create a virtual environment

Installing in a virtual environment is highly recommended.
If you are using `conda`, you can create an environment with the command below, with
"my-env" replaced with any names you'd like.

```shell
conda create -n my-env python  # create environment
conda activate my-env  # enter the virtual environment
```

### Install Rust

In `cylindra`, some computationally intensive functions are implemented in Rust.
You have to install Rust first to run `cylindra`. See the [Rust installation guide](https://www.rust-lang.org/tools/install).

### Install Cylindra

Now, it's ready to install `cylindra`.

```shell
pip install cylindra[all] -U
```

This command will automatically install all the modules `cylindra` depends on, and
compile the Rust codes to be used from Python.

??? info "Avoid installing optional dependencies"

    The "all" option tells `pip` to install everything that will be used in
    `cylindra`. If you don't want all of these, you can manually select the
    dependencies. The optional dependencies are:

    - `pyqt5`
    - `scikit-learn`
    - `mrcfile`
    - `tifffile`
