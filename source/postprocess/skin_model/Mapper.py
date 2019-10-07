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
    def __init__(self, line_structure, structure):
        self.structure = structure
        self.line_structure = line_structure

    def map_line_structure_to_structure(self):
        s_vec = self.line_structure.undeformed[int(self.structure.beam_direction)]
        disp_vec = self.line_structure.displacement
        ang_disp_vec = self.line_structure.angular_displacement

        for i in range(self.structure.num_of_elements):
            mid_p = self._get_element_mid_points(self.structure.elements[i])
            s = mid_p[int(self.structure.beam_direction)]
            self._interpolate_dofs(s, self.structure.elements[i].nodes, s_vec, disp_vec, ang_disp_vec)

    @staticmethod
    def _get_element_mid_points(element):
        mid_x = sum(element.undeformed[0]) / element.num_of_nodes
        mid_y = sum(element.undeformed[1]) / element.num_of_nodes
        mid_z = sum(element.undeformed[2]) / element.num_of_nodes
        return [mid_x, mid_y, mid_z]

    @staticmethod
    def _interpolate_dofs(s, nodes, s_vec, displacement, angular_displacement):
        dx = interpolate_points(s, s_vec, displacement[0])
        dy = interpolate_points(s, s_vec, displacement[1])
        dz = interpolate_points(s, s_vec, displacement[2])
        inter_disp = [dx, dy, dz]

        theta_x = interpolate_points(s, s_vec, angular_displacement[0])
        theta_y = interpolate_points(s, s_vec, angular_displacement[1])
        theta_z = interpolate_points(s, s_vec, angular_displacement[2])
        inter_ang_disp = [theta_x, theta_y, theta_z]

        for node in nodes:
            node.assign_dofs(inter_disp, inter_ang_disp)


def test():
    param = {"length": 100.0, "num_of_elements": 5,
             "geometry": [[0, 15.0, 3.0], [0, 6.0, 9.0], [0, -6.0, 9.0],
                          [0, -15.0, 3.0], [0, -6.0, -9.0], [0, 6.0, -9.0]
                          ],
             "contour_density": 1,
             "record_animation": False,
             "visualize_line_structure": True,
             "beam_direction": "x",
             "scaling_vector": [1.5, 1.0, 2.0],
             "dofs_input": {
                 "x0": [0.0, 25.0, 50.0, 75.0, 100.0],
                 "y0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "z0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "a0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "b0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "g0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "y": [[0.0, 0.1, 0.2, 0.3, 0.4], [0.0, 1.1, 2.2, 3.3, 4.4]],
                 "z": [[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]],
                 "a": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                 "b": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                 "g": [[0.0, 0.0, 0.0, 0.0, np.pi / 6], [0.0, 0.0, 0.0, 0.0, np.pi / 6]],
                 "x": [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]}}

    from source.postprocess.skin_model.StructureModel import Structure
    from source.postprocess.skin_model.LineStructureModel import LineStructure

    s = Structure(param)
    ls = LineStructure(param)
    m = Mapper(ls, s)
    m.map_line_structure_to_structure()


if __name__ == "__main__":
    test()
