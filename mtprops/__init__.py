__version__ = "0.6.6dev0"

from .core import start
from .tomogram import MtTomogram

# TODO: Do not Affine transform twice in reconstruction algorithms
# TODO: Use dask to paralellize reconstruction algorithms
# TODO: Some iterative rotation may be faster using dask, but "rotate" itself uses dask so the effect is trivial. 
# TODO: find seam sometimes fails (no density)