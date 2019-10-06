from scipy.interpolate import CubicSpline
import numpy as np

from source.postprocess.skin_model.NodeModel import Node
from source.postprocess.skin_model.LineStructureModel import LineStructure


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
        self.beam_direction = self.line_structure.beam_direction
        self.interpolated_line_structure = LineStructure()
        self.interpolated_line_structure.beam_direction = self.line_structure.beam_direction
        self.map_line_structure_to_interpolated_line_structure()
        self.map_interpolated_line_structure_to_structure_element()

    def map_interpolated_line_structure_to_structure_element(self):
        for element in self.structure.elements:
            mid_s = sum(element.s_vec) / len(element.s_vec)
            self._interpolate_dofs(mid_s, element.nodes, self.interpolated_line_structure)
        print("Interpolated 3D structure")

    def map_line_structure_to_interpolated_line_structure(self):
        interpolated_line_structure_tmp = LineStructure()
        interpolated_line_structure_tmp.beam_direction = self.interpolated_line_structure.beam_direction

        mid_s_prev = sum(self.structure.elements[0].s_vec) / len(self.structure.elements[0].s_vec)
        for element in self.structure.elements[1:]:
            mid_x, mid_y, mid_z, mid_s = self._get_element_mid_points(element)

            if element != self.structure.elements[-1]:
                s_vec = np.linspace(mid_s_prev, mid_s, INTERPOLATION_DENSITY, endpoint=False)
            else:
                s_vec = np.linspace(mid_s_prev, mid_s, INTERPOLATION_DENSITY)
            for s in s_vec:
                node = Node(mid_x, mid_y, mid_z)
                interpolated_line_structure_tmp.s_vec.append(s)
                interpolated_line_structure_tmp.nodes.append(node)

            mid_s_prev = mid_s

        for i, node in zip(range(len(interpolated_line_structure_tmp.nodes)), interpolated_line_structure_tmp.nodes):
            s = interpolated_line_structure_tmp.s_vec[i]
            # interpolate a node on the interpolated LS to LS
            self._interpolate_dofs(s, node, self.line_structure)
            self._update_dofs(node, interpolated_line_structure_tmp)

        self.interpolated_line_structure = interpolated_line_structure_tmp
        print("Interpolated 1D structure")

    @staticmethod
    def _update_dofs(node, interpolated_line_structure_tmp):
        interpolated_line_structure_tmp.x0_vec.append(node.x0)
        interpolated_line_structure_tmp.y0_vec.append(node.y0)
        interpolated_line_structure_tmp.z0_vec.append(node.z0)
        interpolated_line_structure_tmp.x_vec.append(node.x0)
        interpolated_line_structure_tmp.y_vec.append(node.y0)
        interpolated_line_structure_tmp.z_vec.append(node.z0)
        interpolated_line_structure_tmp.dx_vec.append(node.dx)
        interpolated_line_structure_tmp.dy_vec.append(node.dy)
        interpolated_line_structure_tmp.dz_vec.append(node.dz)
        interpolated_line_structure_tmp.theta_x_vec.append(node.theta_x)
        interpolated_line_structure_tmp.theta_y_vec.append(node.theta_y)
        interpolated_line_structure_tmp.theta_z_vec.append(node.theta_z)

    @staticmethod
    def _get_element_mid_points(element):
        mid_x = sum(element.x_vec) / len(element.x_vec)
        mid_y = sum(element.y_vec) / len(element.y_vec)
        mid_z = sum(element.z_vec) / len(element.z_vec)
        mid_s = sum(element.s_vec) / len(element.s_vec)
        return mid_x, mid_y, mid_z, mid_s

    @staticmethod
    def _interpolate_dofs(s, nodes, line_structure):
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
