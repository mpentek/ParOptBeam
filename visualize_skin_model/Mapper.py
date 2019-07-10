from visualize_skin_model.NodeModel import Node
from visualize_skin_model.LineStructureModel import LineStructure
from scipy.interpolate import CubicSpline
import numpy as np

# curvature - 2nd order deriv
DERIV_ORDER = 2
PRESCRIBED_2ND_ORDER_DERIV = [0, 0]
INTERPOLATION_DENSITY = 5


def interpolate_points(v, x, y):
    interp_x = CubicSpline(x, y, bc_type=(
        (DERIV_ORDER, PRESCRIBED_2ND_ORDER_DERIV[0]),
        (DERIV_ORDER, PRESCRIBED_2ND_ORDER_DERIV[1])))
    # normal = interp_x.derivative()(v)
    return interp_x(v)


class Mapper:
    def __init__(self, structure, line_structure):
        self.structure = structure
        self.line_structure = line_structure
        self.interpolated_line_structure = LineStructure()
        self.interpolated_line_structure.beam_direction = self.line_structure.beam_direction
        self.map_line_structure_to_interpolated_line_structure()
        self.beam_direction = self.line_structure.beam_direction

    def map_interpolated_line_structure_to_structure_floor(self):
        for element in self.structure.elements:
            mid_s = sum(element.s_vec) / len(element.s_vec)

            self.interpolate_dofs(mid_s, element.nodes, self.interpolated_line_structure)

    def map_line_structure_to_interpolated_line_structure(self):
        mid_s_prev = sum(self.structure.elements[0].s_vec) / len(self.structure.elements[0].s_vec)
        for element in self.structure.elements[1:]:
            mid_x = sum(element.x_vec) / len(element.x_vec)
            mid_y = sum(element.y_vec) / len(element.y_vec)
            mid_z = sum(element.z_vec) / len(element.z_vec)
            mid_s = sum(element.s_vec) / len(element.s_vec)

            if element != self.structure.elements[-1]:
                s_vec = np.linspace(mid_s_prev, mid_s, INTERPOLATION_DENSITY, endpoint=False)
            else:
                s_vec = np.linspace(mid_s_prev, mid_s, INTERPOLATION_DENSITY)
            for s in s_vec:
                node = Node(mid_x, mid_y, mid_z)
                # print("NODE: [", mid_x, mid_y, mid_z, "]")
                self.interpolated_line_structure.s_vec.append(s)
                self.interpolated_line_structure.nodes.append(node)

            mid_s_prev = mid_s

        for i, node in zip(range(len(self.interpolated_line_structure.nodes)), self.interpolated_line_structure.nodes):
            s = self.interpolated_line_structure.s_vec[i]
            # interpolate a node on the interpolated LS to LS
            self.interpolate_dofs(s, node, self.line_structure)
            self.interpolated_line_structure.x_vec.append(node.x0)
            self.interpolated_line_structure.y_vec.append(node.y0)
            self.interpolated_line_structure.z_vec.append(node.z0)
            self.interpolated_line_structure.dx_vec.append(node.dx)
            self.interpolated_line_structure.dy_vec.append(node.dy)
            self.interpolated_line_structure.dz_vec.append(node.dz)
            self.interpolated_line_structure.theta_x_vec.append(node.theta_x)
            self.interpolated_line_structure.theta_y_vec.append(node.theta_y)
            self.interpolated_line_structure.theta_z_vec.append(node.theta_z)

    @staticmethod
    def interpolate_dofs(s, nodes, line_structure):
        dx = interpolate_points(s, line_structure.s_vec, line_structure.dx_vec)
        dy = interpolate_points(s, line_structure.s_vec, line_structure.dy_vec)
        dz = interpolate_points(s, line_structure.s_vec, line_structure.dz_vec)

        theta_x = interpolate_points(s, line_structure.s_vec, line_structure.theta_x_vec)
        theta_y = interpolate_points(s, line_structure.s_vec, line_structure.theta_y_vec)
        theta_z = interpolate_points(s, line_structure.s_vec, line_structure.theta_z_vec)

        if isinstance(nodes, (list,)):
            for node in nodes:
                node.assign_dofs(dx, dy, dz, theta_x, theta_y, theta_z)
        else:
            nodes.assign_dofs(dx, dy, dz, theta_x, theta_y, theta_z)
