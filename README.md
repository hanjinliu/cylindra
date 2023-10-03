# cylindra

`cylindra` is a GUI-integrated cryo-ET image analysis tool for cylindric periodic structures such as microtubules.

![](resources/fig.png)

## Installation

In `cylindra`, computationally-intensive part is implemented in Rust. You have to install Rust first to run `cylindra`. See the [Rust installation guide](https://www.rust-lang.org/tools/install).

- From source

```shell
git clone git+https://github.com/hanjinliu/cylindra
```

To install using pip, run following commands under your desired environment.

```shell
cd cylindra
pip install -e .
```

## Usage

- From shell

  ```shell
  cylindra
  ```

- From a Python interpreter

  ```python
  from cylindra import start

  # launch a napari viewer with a cylindra dock widget.
  ui = start()
  ```

## Implemented Functions

- Automatic/manual fitting of splines to cylindrical structures in 3D.
- Calculating structural parameters (such as radius, lattice spacing and skew angle) using Cylindric Fourier transformation.
- Automatic/manual determination of polarity, protofilament number etc.
- Monomer mapping along splines.
- Classical subtomogram averaging and alignment.
- Microtubule seam search with or without binding proteins.
- Subtomogram alignment with 1D constraint using Viterbi algorithm.
- Subtomogram alignment with 2D constraint using simulated annealing optimization.
- Tomogram simulation of cylindric structure.

## Prerequisite and Recommendations

- **Python &ge; 3.10**. This project follows [spec-0000](https://scientific-python.org/specs/spec-0000/).
- **Sufficient memory size**. Most of the intense calculations are done out-of-core using `dask`, so that you can even run on 8-GB memory PC in many cases. However, larger memory size will make parallel processing more efficient. &ge;32 GB is recommended.
- **Images should be loaded from SSD**. Raw image stacks are loaded lazily in most of the processes. Loading from HDD will slow down many analyses as well.

## Issues

If you encountered any bugs or have any requests, feel free to [report an issue](https://github.com/hanjinliu/cylindra/issues). We'll appreciate if you find some methods are over-fitted to microtubules and do not work well on other cylindric structures.

For better reproducibility, please copy your environments from `Others > cylindra info` and the recorded macro from
`Others > Macro > Show macro`.

## Citation

If you find `cylindra` useful in your work, please consider citing the following paper.

```
TODO
```
