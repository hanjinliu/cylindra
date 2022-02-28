from mtprops import MtTomogram
from mtprops.const import Ori
from pathlib import Path

def test_cache_in_tomogram():
    path = Path(__file__).parent / "13pf_MT.tif"
    tomo = MtTomogram.imread(path, scale=1.052, light_background=False)
    tomo.add_spline([[18.97, 190.0, 28.99], 
                     [18.97, 107.8, 51.48],
                     [18.97,  35.2, 79.90]])
    spl = tomo.splines[0]
    
    assert spl.radius is None
    assert spl.localprops is None
    assert spl.globalprops is None
    assert spl.orientation == Ori.none
    
    tomo.set_radius(0)
    tomo.local_ft_params(0)
    tomo.global_ft_params(0)
    
    spl.orientation = Ori.PlusToMinus
    
    assert spl.radius is not None
    assert spl.localprops is not None
    assert spl.globalprops is not None
    assert spl.orientation == Ori.PlusToMinus
    
    tomo.make_anchors(0)
    
    assert spl.radius is not None
    assert spl.localprops is None
    assert spl.globalprops is not None
    assert spl.orientation == Ori.PlusToMinus
    
    tomo.local_ft_params(0)
    tomo.fit(0)
    
    assert spl.radius is None
    assert spl.localprops is None
    assert spl.globalprops is None
    assert spl.orientation == Ori.none
    
    # test straightening
    assert spl.cart_stimg is None
    assert spl.cyl_stimg is None
    tomo.set_radius(0)
    tomo.straighten_cylindric(0)
    
    assert spl.cart_stimg is None
    assert spl.cyl_stimg is not None
    
    tomo.straighten(0)
    
    assert spl.cart_stimg is not None
    assert spl.cyl_stimg is not None
    
    out = tomo.straighten(0, size=6)
    
    assert out.shape != spl.cart_stimg.shape
    assert spl.cart_stimg is not None
    