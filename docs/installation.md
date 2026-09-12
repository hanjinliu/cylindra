# Installation

### Create a virtual environment

Installing in a virtual environment is highly recommended.
If you are using `conda`, you can create an environment with the command below, with
"my-env" replaced with any names you'd like.

```shell
conda create -n my-env python  # create environment
conda activate my-env  # enter the virtual environment
```

### Install Cylindra

Now, it's ready to install `cylindra`. Following command will install `cylindra` and all
the relevant dependencies in the current virtual environment.

=== "From PyPI"

    ```shell
    pip install cylindra[recommended] -U
    ```

    If you are using `venv` from `uv`,

    ```shell
    uv pip install cylindra[recommended] -U
    ```

=== "From napari hub"

    `cylindra` is available as a [napari plugin](https://napari-hub.org/plugins/cylindra).
    You can install it from the napari plugin manager.

    1. Launch `napari`.
    2. Click `Plugins > Install/Uninstall plugins...`.
    3. Type "cylindra" in the filter box. Click the "Install" button.

=== "From the GitHub main branch"

    ```shell
    pip install git+https://github.com/hanjinliu/cylindra.git
    ```

=== "Build from the source"

    Clone the repository. Install [`git`](https://git-scm.com) and run:

    ```shell
    git clone https://github.com/hanjinliu/cylindra
    ```

    or manually download the repository as a ZIP file and extract it.

    Compile Rust code. You will nedd to have [`Rust`](https://www.rust-lang.org/learn/get-started) installed.

    ```shell
    cd cylindra
    pip install .[recommended]
    ```

    For Windows, you may need to install the [Visual Studio](https://visualstudio.microsoft.com/) with the "Desktop development with C++" workload.

??? info "Avoid installing optional dependencies"

    The "recommended" option tells `pip` to install packages that works well.
    If you don't want all of these, you can manually select the dependencies.

    - `pip install cylindra` ... minimum installation with only the essential dependencies.
    - `pip install cylindra[all]` ... all optional dependencies for `cylindra`.
    - `pip install cylindra[recommended]` ... all optional dependencies with recommended versions.

You can check if the installation succeeded by running `cylindra -v`.
