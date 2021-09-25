
class Header:
    label = "label"
    number = "number"
    z = "z"
    y = "y"
    x = "x"
    angle_zy = "angle_zy"
    angle_yx = "angle_yx"
    MTradius = "MTradius"
    curvature = "curvature"
    pitch = "pitch"
    nPF = "nPF"
    spl_knot_vec = "spl_knot_vec"
    spl_coeff_z = "spl_coeff_z"
    spl_coeff_y = "spl_coeff_y"
    spl_coeff_x = "spl_coeff_x"
    spl_u = "spl_u"
    
    @classmethod
    def zyx(cls):
        return [cls.z, cls.y, cls.x]