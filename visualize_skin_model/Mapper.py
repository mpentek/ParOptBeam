from visualize_skin_model.NodeModel import Node
from visualize_skin_model.LineStructureModel import LineStructure
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import numpy as np

# curvature - 2nd order deriv
DERIV_ORDER = 2
PRESCRIBED_2ND_ORDER_DERIV = [0, 0]
INTERPOLATION_DENSITY = 5


def interpolate_points_np(v, x, y):
    if v <= x[0]:
        return y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (v - x[0])
    elif v >= x[-1]:
        return y[-2] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (v - x[-2])
    else:
        f = interp1d(x, y, kind='cubic')
        return f(v)


def interpolate_points_scipy(x, f):
    interp_x = CubicSpline(x, f, bc_type=(
        (DERIV_ORDER, PRESCRIBED_2ND_ORDER_DERIV[0]),
        (DERIV_ORDER, PRESCRIBED_2ND_ORDER_DERIV[1])))
    return interp_x


class Mapper:
    def __init__(self, structure, line_structure):
        self.structure = structure
        self.line_structure = line_structure
        self.interpolated_line_structure = LineStructure()

    def map_interpolated_line_structure_to_structure_floor(self):
        for floor in self.structure.floors:
            mid_z = sum(floor.z_vec) / len(floor.z_vec)

            self.interpolate_dofs(mid_z, floor.nodes, self.line_structure)

    def map_line_structure_to_interpolated_line_structure(self):
        mid_z_prev = 0
        for floor in self.structure.floors:
            mid_x = sum(floor.x_vec) / len(floor.x_vec)
            mid_y = sum(floor.y_vec) / len(floor.y_vec)
            mid_z = sum(floor.z_vec) / len(floor.z_vec)

            if mid_z != mid_z_prev:
                z_vec = np.linspace(mid_z_prev, mid_z, INTERPOLATION_DENSITY)
            else:
                z_vec = [mid_z]
            for z in z_vec:
                node = Node(mid_x, mid_y, z)
                self.interpolate_dofs(z, node, self.line_structure)

                self.interpolated_line_structure.nodes.append(node)
                self.interpolated_line_structure.x_vec.append(node.x0)
                self.interpolated_line_structure.y_vec.append(node.y0)
                self.interpolated_line_structure.z_vec.append(node.z0)

            mid_z_prev = mid_z

        self.interpolated_line_structure.print_nodal_infos()

    @staticmethod
    def interpolate_dofs(z, nodes, line_structure):
        dx = interpolate_points_np(z, line_structure.z_vec, line_structure.dx_vec)
        dy = interpolate_points_np(z, line_structure.z_vec, line_structure.dy_vec)
        dz = interpolate_points_np(z, line_structure.z_vec, line_structure.dz_vec)
        theta_x = interpolate_points_np(z, line_structure.z_vec, line_structure.theta_x_vec)
        theta_y = interpolate_points_np(z, line_structure.z_vec, line_structure.theta_y_vec)
        theta_z = interpolate_points_np(z, line_structure.z_vec, line_structure.theta_z_vec)

        if isinstance(nodes, (list,)):
            for node in nodes:
                node.add_dofs(dx, dy, dz, theta_x, theta_y, theta_z)
        else:
            nodes.add_dofs(dx, dy, dz, theta_x, theta_y, theta_z)
