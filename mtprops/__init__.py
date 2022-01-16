__version__ = "0.7.0.dev0"
__author__ = "Hanjin Liu"
__email__ = "liuhanjin-sc@g.ecc.u-tokyo.ac.jp"

from .core import start
from .tomogram import MtTomogram
from .widget import MTPropsWidget

# TODO: Do not Affine transform twice in reconstruction algorithms
# TODO: Use dask to paralellize reconstruction algorithms
# TODO: Some iterative rotation may be faster using dask, but "rotate" itself uses dask so the effect is trivial. 
# TODO: find seam sometimes fails (no density)