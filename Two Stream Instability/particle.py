class Particle:
    def __init__(self, x, v, domain, spline_order: int = 1, weight_factor=1.0, electric_field=0, color=None):
        self.domain = domain
        self.x_true = x
        self.v = v
        self.spline_order = spline_order
        self.weight_factor = weight_factor
        self.electric_field = electric_field
        self.color = color if color != None else 'c'

    @property
    def x(self):
        return (self.x_true - self.domain[0]) % (self.domain[1] - self.domain[0]) + self.domain[0]

    def get_area_fraction(self, x, dx):
        x_ref = abs(self.x - x) / dx
        match self.spline_order:
            case 0:
                if x_ref < 1:
                    return 1 - x_ref
                return 0
            case 1:
                if x_ref <= 0.5:
                    return 0.75-x_ref**2
                if x_ref <= 1.5:
                    return 0.5*(1.5-x_ref)**2
                return 0
            case _:
                if x_ref < 1:
                    return 1 - x_ref
                return 0