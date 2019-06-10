from NodeModel import Node
from StructureModel import Structure, Floor
from LineStructureModel import LineStructure
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

# curvature - 2nd order deriv
DERIV_ORDER = 2
PRESCRIBED_2ND_ORDER_DERIV = [0, 0]


def extrap(v, x, y):
    if v <= x[0]:
        return y[0]+(y[1]-y[0])/(x[1]-x[0])*(v-x[0])
    elif v >= x[-1]:
        return y[-2]+(y[-1]-y[-2])/(x[-1]-x[-2])*(v-x[-2])
    else:
        f = interp1d(x, y, kind='cubic') 
        return f(v)

class Mapper:
    def __init__(self, structure, line_structure):
        self.structure = structure
        self.line_structure = line_structure

    def interpolate_points(self):
        interp_x = CubicSpline(self.x_vec, self.z_vec, bc_type=(
                          (DERIV_ORDER, PRESCRIBED_2ND_ORDER_DERIV[0]),
                          (DERIV_ORDER, PRESCRIBED_2ND_ORDER_DERIV[1])))

    def map_line_structure_to_frame_mid_point(self):
        for floor in self.structure.floors:
            mid_x = sum(floor.x_vec) / len(floor.x_vec)
            mid_y = sum(floor.y_vec) / len(floor.y_vec)
            mid_z = sum(floor.z_vec) / len(floor.z_vec)
            node = Node(mid_x, mid_y, mid_z)

            dx = extrap(mid_z, self.line_structure.z_vec, self.line_structure.dx_vec)
            dy = extrap(mid_z, self.line_structure.z_vec, self.line_structure.dy_vec)
            dz = extrap(mid_z, self.line_structure.z_vec, self.line_structure.dz_vec)
            theta_x = extrap(mid_z, self.line_structure.z_vec, self.line_structure.theta_x_vec)
            theta_y = extrap(mid_z, self.line_structure.z_vec, self.line_structure.theta_y_vec)
            theta_z = extrap(mid_z, self.line_structure.z_vec, self.line_structure.theta_z_vec)
            
            node.add_dofs(dx, dy, dz, theta_x, theta_y, theta_z)
