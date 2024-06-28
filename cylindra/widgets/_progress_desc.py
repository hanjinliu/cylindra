from __future__ import annotations

from napari.layers import Layer

from cylindra.const import ImageFilter


def _get_name(layer) -> str:
    if isinstance(layer, str):
        return layer
    elif isinstance(layer, Layer):
        return layer.name
    elif isinstance(layer, list):
        if len(layer) == 0:
            return "no layer"
        elif len(layer) == 1:
            return _get_name(layer[0])
        else:
            return f"{_get_name(layer[0])} etc."
    return repr(layer)


# functions
def fmt_layer(fmt: str):
    """Define a formatter for progressbar description."""
    return lambda layer: fmt.format(_get_name(layer))


def fmt_layers(fmt: str):
    """Define a formatter for progressbar description."""

    def _formatter(layers: list[Layer]):
        if len(layers) == 1:
            return fmt.format(_get_name(layers[0]))
        return fmt.format(f"{len(layers)} layers")

    return _formatter


def filter_image_fmt(method: ImageFilter):
    return f"Running {ImageFilter(method).name} filter"


def align_averaged_fmt(layers: list[Layer]):
    n = len(layers)
    total = 2 * n + 1
    yield f"(0/{total}) Preparing template images for alignment"
    for i in range(n):
        name = _get_name(layers[i])
        yield f"({i * 2 + 1}/{total}) Subtomogram averaging of {name!r}"
        yield f"({i * 2 + 2}/{total}) Aligning template to the average image of {name!r}"
    yield f"({total}/{total}) Finishing"


def align_all_fmt(layers: list[Layer]):
    name = _get_name(layers)
    yield f"(0/2) Preparing template images for {name!r}"
    yield f"(1/2) Alignment of {name!r}"


def align_template_free_fmt(layers: list[Layer]):
    name = _get_name(layers)
    yield f"(0/3) Preparing template images for {name!r}"
    yield f"(1/3) Averaging subtomograms of {name!r}"
    yield f"(2/3) Aligning subtomograms of {name!r}"
    yield "(4/4) Finishing"


def align_viterbi_fmt(layer: Layer):
    name = _get_name(layer)
    yield f"(0/3) Preparing template images for {name!r}"
    yield f"(1/3) Calculating the correlation landscape of {name!r}"
    yield f"(2/3) Running Viterbi alignment of {name!r}"
    yield "(3/3) Finishing"


def align_annealing_fmt(layer: Layer):
    name = _get_name(layer)
    yield f"(0/3) Preparing template images for {name!r}"
    yield f"(1/3) Calculating the correlation landscape of {name!r}"
    yield f"(2/3) Running Annealing of {name!r}"
    yield "(3/3) Finishing"


def construct_landscape_fmt(layer: Layer):
    name = _get_name(layer)
    yield f"(0/3) Preparing template images for {name!r}"
    yield f"(1/3) Calculating the correlation landscape of {name!r}"
    yield f"(2/3) Building the surface of {name!r}"
    yield "(3/3) Finishing"


def classify_pca_fmt(layer: Layer):
    name = _get_name(layer)
    yield f"(0/4) Creating template image for PCA clustering from {name!r}"
    yield "(1/4) Fitting PCA model"
    yield "(2/4) Transforming all the images"
    yield "(3/4) Creating average images for each cluster"
    yield "(4/4) Finishing"
