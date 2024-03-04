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

Now, it's ready to install `cylindra`.

=== "Using `pip`"

    Following command will install `cylindra` and all the relevant dependencies in the
    current virtual environment.

    ```shell
    pip install cylindra[all] -U
    ```

=== "From the GitHub main branch"

    Following command will install `cylindra` from the main branch of the GitHub
    repository.

    ```shell
    pip install git+https://github.com/hanjinliu/cylindra.git
    ```

=== "Build from the source"

    You can clone the repository and build locally. This option requires `git` and
    the Rust programming language. Following commands will save all the files in the
    "cylindra" folder, compile the Rust files and install them.

    ```shell
    git clone https://github.com/hanjinliu/cylindra
    cd cylindra
    pip install .[all]
    ```

??? info "Avoid installing optional dependencies"

    The "all" option tells `pip` to install everything that will be used in
    `cylindra`. If you don't want all of these, you can manually select the
    dependencies. The optional dependencies are:

    - `pyqt5` ... This is the default GUI backend. You can also use `pyqt6` or
      `pyside6`.
    - `scikit-learn` ... Conventional machine learning library. Princilple component
      analysis (PCA) and k-means clustering need this library.
    - `mrcfile` ... A library for reading and writing MRC files. This is needed if you
      want to use MRC files.
    - `tifffile` ... A library for reading and writing TIFF files. This is needed if you
      want to use TIFF files.

You can check if the installation succeeded by running `cylindra -v`.
