from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari.layers import Layer


# functions
def fmt_layer(fmt: str):
    """Define a formatter for progressbar description."""

    def _formatter(layer: "Layer"):
        return fmt.format(layer.name)

    return _formatter


def fmt_layers(fmt: str):
    """Define a formatter for progressbar description."""

    def _formatter(layers: "list[Layer]"):
        if len(layers) == 1:
            return fmt.format(repr(layers[0].name))
        return fmt.format(f"{len(layers)} layers")

    return _formatter


def align_averaged_fmt(layers: "list[Layer]"):
    n = len(layers)
    total = 2 * n + 1
    yield f"(0/{total}) Preparing template images for alignment"
    for i in range(n):
        name = layers[i].name
        yield f"({i * 2 + 1}/{total}) Subtomogram averaging of {name!r}"
        yield f"({i * 2 + 2}/{total}) Aligning template to the average image of {name!r}"
        yield f""  # The actual yield statement is in the function
    yield f"({total}/{total}) Finishing"


def align_template_free_fmt():
    yield "(0/4) Caching subtomograms of"
    yield "(1/4) Preparing template images for"
    yield "(2/4) Averaging subtomograms of"
    yield "(3/4) Aligning subtomograms of"
    yield "(4/4) Finishing"


def align_viterbi_fmt(layer: "Layer"):
    yield f"(0/3) Preparing template images for {layer.name!r}"
    yield f"(1/3) Calculating the correlation landscape of {layer.name!r}"
    yield f"(2/3) Running Viterbi alignment of {layer.name!r}"
    yield "(3/3) Finishing"


def align_annealing_fmt(layer: "Layer"):
    yield f"(0/3) Preparing template images for {layer.name!r}"
    yield f"(1/3) Calculating the correlation landscape of {layer.name!r}"
    yield f"(2/3) Running Annealing of {layer.name!r}"
    yield "(3/3) Finishing"


def classify_pca_fmt(layer: "Layer"):
    yield f"(0/5) Caching subtomograms of {layer.name!r}"
    yield "(1/5) Creating template image for PCA clustering"
    yield "(2/5) Fitting PCA model"
    yield "(3/5) Transforming all the images"
    yield "(4/5) Creating average images for each cluster"
    yield "(5/5) Finishing"
