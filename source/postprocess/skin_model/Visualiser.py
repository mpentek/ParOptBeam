import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import animation
from os.path import join
import numpy as np

from source.postprocess.skin_model.Mapper import Mapper

# plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg: /usr/bin/ffmpeg'
plt.style.use('classic')


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Visualiser:
    def __init__(self, line_structure, structure):
        self.line_structure = line_structure
        self.structure = structure
        self.mapper = Mapper(line_structure, structure)
        self.scale = 2e5
        self.steps = self.structure
        self.is_record_animation = self.structure.is_record_animation
        self.is_visualize_line_structure = self.structure.is_visualize_line_structure
        self.mode = self.structure.mode
        self.frequency = self.structure.frequency
        self.period = self.structure.period
        self.steps = self.line_structure.steps
        self.frame_time = np.linspace(0, self.period, self.steps)
        self.result_path = self.structure.result_path
        self.plot_title = self.structure.plot_title

        self.line_structure.apply_transformation_for_line_structure()
        self.structure.apply_transformation_for_structure()

        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(
            111, projection='3d', aspect='equal', azim=-60, elev=10)

        self.animate()

    def init(self):
        self.fig.suptitle(self.plot_title)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.set_coordinate_in_real_size()
        self.visualize_coordinate()

    def visualize_coordinate(self):
        arrow_prop_dict = dict(
            mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
        a = Arrow3D([0, 10], [0, 0], [0, 0], **arrow_prop_dict, color='r')
        self.ax.add_artist(a)
        a = Arrow3D([0, 0], [0, 10], [0, 0], **arrow_prop_dict, color='b')
        self.ax.add_artist(a)
        a = Arrow3D([0, 0], [0, 0], [0, 10], **arrow_prop_dict, color='g')
        self.ax.add_artist(a)

        self.ax.text(0.0, 0.0, -1, r'$o$')
        self.ax.text(11, 0, 0, r'$x$')
        self.ax.text(0, 11, 0, r'$y$')
        self.ax.text(0, 0, 11, r'$z$')

    def set_coordinate_in_real_size(self):
        X, Y, Z = [], [], []

        for frame in self.structure.frames:
            X += frame.deformed[0].tolist()
            Y += frame.deformed[1].tolist()
            Z += frame.deformed[2].tolist()

        mid_x = (max(X) + min(X)) * 0.5
        mid_y = (max(Y) + min(Y)) * 0.5
        mid_z = (max(Z) + min(Z)) * 0.5
        max_range = np.array(
            [max(X) - min(X), max(Y) - min(Y), max(Z) - min(Z)]).max() / 2.0
        self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        self.visualize_coordinate()

    def scale_deformation(self, obj):
        x_vec = obj.undeformed[0] + \
                np.subtract(obj.deformed[0], obj.undeformed[0]) * self.scale
        y_vec = obj.undeformed[1] + \
                np.subtract(obj.deformed[1], obj.undeformed[1]) * self.scale
        z_vec = obj.undeformed[2] + \
                np.subtract(obj.deformed[2], obj.undeformed[2]) * self.scale
        return x_vec, y_vec, z_vec

    def visualise_line_structure(self):
        x_vec, y_vec, z_vec = self.scale_deformation(self.line_structure)

        z = np.array([z_vec, z_vec])
        self.ax.plot_wireframe(x_vec, y_vec, z, color='b', linewidth=3)
        for node in self.line_structure.nodes:
            x = node.undeformed[0] + (node.deformed[0] - node.undeformed[0]) * self.scale
            y = node.undeformed[1] + (node.deformed[1] - node.undeformed[1]) * self.scale
            z = node.undeformed[2] + (node.deformed[2] - node.undeformed[2]) * self.scale
            self.ax.scatter(x, y, z, marker='o', c='r', s=50)

    def visualise_structure(self):
        self.visualise_element()
        self.visualize_frame()

    def visualise_element(self):
        for element in self.structure.elements:
            x_vec, y_vec, z_vec = self.scale_deformation(element)

            x_vec = np.append(x_vec, x_vec[0])
            y_vec = np.append(y_vec, y_vec[0])
            z_vec = np.append(z_vec, z_vec[0])

            z = np.array([z_vec, z_vec])

            self.ax.plot_wireframe(x_vec, y_vec, z, color='black')

    def visualize_frame(self):
        for frame in self.structure.frames:
            x_vec, y_vec, z_vec = self.scale_deformation(frame)

            z = np.array([z_vec, z_vec])
            self.ax.plot_wireframe(x_vec, y_vec, z, color='black')

    def animate(self):
        # TODO add dependency on ffmpeg somewhere - need to install ffmpeg for users
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.steps/self.period, metadata=dict(artist='Me'), bitrate=1800)
        a = animation.FuncAnimation(self.fig,
                                    self.update,
                                    frames=self.steps,
                                    init_func=self.init(),
                                    repeat=True)
        if self.is_record_animation:
            file = join(self.result_path,  'mode_' + self.mode + '_skin_model.mp4')
            a.save(file, writer)
            print("Successfully written animation video!")

        plt.grid()
        plt.tight_layout()
        plt.show()

    def update(self, step):
        """
        plotting the deformed structure with respect to the displacement
        """
        wframe = self.fig.gca()
        if wframe is not None:
            self.ax.cla()
        print("time: " + str(step))
        self.line_structure.update_dofs(step)
        self.mapper.map_line_structure_to_structure()

        self.line_structure.apply_transformation_for_line_structure()
        self.structure.apply_transformation_for_structure()

        self.visualise_structure()
        if self.is_visualize_line_structure:
            self.visualise_line_structure()
        self.set_coordinate_in_real_size()
        self.ax.text(0, 5, 15, '{0:.2f}'.format(self.frame_time[step]) + "[s]", fontsize=20, color='red')


def test():
    param = {"length": 100.0,
             "geometry": [[0, 15.0, 3.0], [0, 6.0, 9.0], [0, -6.0, 9.0],
                          [0, -15.0, 3.0], [0, -6.0, -9.0], [0, 6.0, -9.0]
                          ],
             "contour_density": 1,
             "record_animation": True,
             "visualize_line_structure": True,
             "beam_direction": "x",
             "scaling_vector": [1.0, 1.0, 1.0],
             'result_path': '.',
             'mode': '1',
             'frequency': 0.1,
             'period': 4.0,
             "dofs_input": {
                 "x0": [0.0, 25.0, 50.0, 75.0, 100.0],
                 "y0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "z0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "a0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "b0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "g0": [0.0, 0.0, 0.0, 0.0, 0.0],
                 "y": [[0.2, 0.1], [0.3, 0.2], [0.4, 0.3], [0.5, 0.4], [0.4, 0.5]],
                 "z": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [4.0, 0.0]],
                 "a": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                 "b": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                 "g": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [np.pi/12, np.pi/15]],
                 "x": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]}}
    from source.postprocess.skin_model.StructureModel import Structure
    from source.postprocess.skin_model.LineStructureModel import LineStructure

    s = Structure(param)
    ls = LineStructure(param)
    v = Visualiser(ls, s)


if __name__ == "__main__":
    test()
