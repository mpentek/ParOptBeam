import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import *
from matplotlib import animation

####################################################################################
# @para:
#  dx: displacement in x direction
#  dy: displacement in y direction
#  phi: rotation around axis x
#  beta: rotation around axis y
#  alpha: torstion
#  num_of_points_in_height: density of discretization in height
####################################################################################

fig = plt.figure()
#ax = fig.add_subplot(111, projection = '3d', animation = True)
ax = Axes3D(fig)
fig.set_size_inches(5, 10, forward=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('height')

class Structure:
    def __init__(self, 
                 displacement, 
                 rotation, 
                 num_of_points_in_height, 
                 n_floors, floor_geometry, floor_height, 
                 densify=False):

        self.n_floors = n_floors
        self.dx = displacement["dx"]
        self.dy = displacement["dy"]
        self.phi = rotation["phi"]
        self.beta = rotation["beta"]
        self.alpha = rotation["alpha"]
        self.num_of_points_in_height = num_of_points_in_height
        self.floors = []
        self.floor_height = floor_height
        self.height = self.n_floors * self.floor_height
        self.floor_geometry = floor_geometry

        if densify:
            self._densify_contour()


    def _densify_contour(self, parts=5):
        new_floor_geometry = {}
        new_floor_geometry["points"] = []
        for i in range(len(self.floor_geometry["points"])):
            new_floor_geometry["points"].append(self.floor_geometry["points"][i])

            new_floor_geometry["points"] += self._get_equidistant_points(
                    [self.floor_geometry["points"][i % len(self.floor_geometry["points"])]["x"],
                    self.floor_geometry["points"][i % len(self.floor_geometry["points"])]["y"]],
                    [self.floor_geometry["points"][(i+1) % len(self.floor_geometry["points"])]["x"],
                    self.floor_geometry["points"][(i+1) % len(self.floor_geometry["points"])]["y"]],
                    parts)

        self.floor_geometry["points"] = new_floor_geometry["points"]


    def _get_equidistant_points(self, p1, p2, parts):
        return [{"x": val1, "y": val2} for val1, val2 in zip(np.linspace(p1[0], p2[0], parts+1),
                                                             np.linspace(p1[1], p2[1], parts+1))]


    def _reconstruct_function_in_vectors(self, t):
        self.undeformed_z = np.linspace(0, self.height, self.num_of_points_in_height) # the z vector for eigenvalue
        self.vec_dx = []
        self.vec_dy = []
        self.vec_phi = []    # roation-x
        self.vec_beta = []   # roation-y
        self.vec_alpha = []  # roation-z
        
        for z in self.undeformed_z:
            temporal_displacement_x = self.dx(t, z)
            self.vec_dx.append( temporal_displacement_x )
            displacement_y = self.dy(t, z)
            self.vec_dy.append( displacement_y )
            rotation_phi = self.phi(t, z)
            self.vec_phi.append( rotation_phi )
            rotation_beta = self.beta(t, z)
            self.vec_beta.append( rotation_beta )
            rotation_alpha = self.alpha(t, z)
            self.vec_alpha.append( rotation_alpha )


    def _create_all_floors(self, t):
        """
        This function creates the entire building
        The z-vector is discretized with user desired sampling size 
        """
        # line structure will be constructed first as the angle 
        # and surface normal of each floor is derived from the line structure
        self.plot_1d_structure(t)
        self.floors = []
        
        for i in range(0, self.n_floors + 1):
            # the interplotation takes a certain value x0 out of a function f(x),
            # here we have the displacement functions dx(z) and dy(z), 
            # the values at the floor height will be interpolated
            floor_displacement_x = np.interp( i*self.floor_height, self.undeformed_z, self.vec_dx)
            floor_displacement_y = np.interp( i*self.floor_height, self.undeformed_z, self.vec_dy)
            theta = np.interp( i*self.floor_height, self.undeformed_z, self.rotation_angles)
            print("floor_displacement_x: ", floor_displacement_x)
            print("floor_displacement_y: ", floor_displacement_y)
            #theta2 = np.interp( (i-1)*self.floor_height, self.undeformed_z, self.rotation_angles)
            #theta = (theta1 + theta2)/2
            # apply displacement for floor
            dx = np.interp( i*self.floor_height, self.undeformed_z, np.squeeze( np.asarray(self.displacement_vector[:, 0])))
            dy = np.interp( i*self.floor_height, self.undeformed_z, np.squeeze( np.asarray(self.displacement_vector[:, 1])))
            dz = np.interp( i*self.floor_height, self.undeformed_z, np.squeeze( np.asarray(self.displacement_vector[:, 2])))
            print("dx = ", dx)
            print("dy = ", dy)
            # rotational axis u = [ux, uy, uz]
            ux = np.interp( i*self.floor_height, self.undeformed_z, np.squeeze( np.asarray(self.surface_normals[:, 0])))
            uy = np.interp( i*self.floor_height, self.undeformed_z, np.squeeze( np.asarray(self.surface_normals[:, 1])))
            uz = np.interp( i*self.floor_height, self.undeformed_z, np.squeeze( np.asarray(self.surface_normals[:, 2])))
            print("[ux, uy, uz] = ", [ux, uy, uz])
            #rotation_axis =  np.interp( i*self.floor_height, self.undeformed_z, self.surface_normals)
            self._create_single_floor(i, floor_displacement_x, floor_displacement_y, theta, dx, dy, dz, ux, uy, uz)


    def _create_single_floor(self, number_of_floor, displacement_x, displacement_y,  theta, dx, dy, dz, ux, uy, uz):
        """
        creating single floor based on the given floor geometry
        """
        floor = []
        z0 = number_of_floor * self.floor_height
        print("***************************NEW FLOOR*********************************")
        print(theta)
        for point in self.floor_geometry["points"]:
            x0 = point["x"]
            y0 = point["y"]
            print("x: ", str(x0))
            print("y: ", str(y0))
            print("z: ", str(z0))
            x, y, z = self._apply_rotation_for_floor(x0, y0, z0, theta, 0.0, 1.0, 0.0)
            print("x: ", str(x))
            print("y: ", str(y))
            print("z: ", str(z))
            print("==================================")
            #x, y, z = self._apply_displacement_for_floor(x, y, z, dx, dy, dz) 
            x += displacement_x
            y += displacement_y
            #x, y = self._apply_torsion(x, y, floor_alpha)
            node = Node(x, y, z)
            floor.append(node)
        self.floors.append(floor)


    def _apply_torsion(self, x0, y0, floor_alpha):
        x = cos(floor_alpha) * x0 - sin(floor_alpha) * y0
        y = sin(floor_alpha) * x0 + cos(floor_alpha) * y0
        return x, y


    def _apply_rotation_for_line_structure(self, x0, y0, z0, floor_phi, floor_beta, floor_alpha):
        # change the data type from np.float to float
        floor_phi   = float(floor_phi)
        floor_beta  = float(floor_beta)
        floor_alpha = float(floor_alpha)
        # rotation matrix around axis x
        Rx = np.matrix([[1,     0,              0              ],
                        [0,     cos(floor_phi), -sin(floor_phi)],
                        [0,     sin(floor_phi), cos(floor_phi) ]])

        # rotation matrix around axis y
        Ry = np.matrix([[cos(floor_beta),   0,    sin(floor_beta)],
                        [0,                 1,    0              ],
                        [-sin(floor_beta),  0,    cos(floor_beta)]])

       # rotation matrix around axis z
        Rz = np.matrix([[cos(floor_alpha), -sin(floor_alpha),  0],
                        [sin(floor_alpha), cos(floor_alpha),   0],
                        [0,                0,                  1]])

        previous_coordinate = np.matrix([[x0],[y0],[z0]])
        new_coordinate = (Ry*Rx)*previous_coordinate
        x = float(new_coordinate[0][0])
        y = float(new_coordinate[1][0])
        z = float(new_coordinate[2][0])
        return x, y, z


    def _return_translation_matrix(self, dx, dy, dz):
        # translation matrix to move a point by [dx, dy, dz]
        T = np.matrix([[1,   0,    0,    dx],
                       [0,   1,    0,    dy],
                       [0,   0,    1,    dz],
                       [0,   0,    0,     1]])
        return T


    def _apply_rotation_for_floor(self, x0, y0, z0, theta, ux, uy, uz):
        c = cos(theta)
        s = sin(theta)
        print("theta: ", theta*180/3.1415926)
        # rotation matrix around axis u = [ ux, uy, uz]
        R = np.matrix([[c+ ux*ux*(1-c),   ux*uy*(1-c)-uz*s,  ux*uz*(1-c)+uy*s,  0],
                       [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c),     uy*uz*(1-c)-ux*s,  0],
                       [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s,  c+uz*uz*(1-c),     0],
                       [0,                 0,                   0,              1]])

        print(R)
        previous_coordinate = np.matrix([[x0],[y0],[z0],[0.0]])
        #T1 = self._return_translation_matrix(-x0, -y0, -z0)
        #T2 = self._return_translation_matrix(x0, y0, z0)
        #print(T1)
        #print(T2)
        #new_coordinate = T2*R*T1*previous_coordinate
        new_coordinate = R*previous_coordinate
        
        #print(T1*previous_coordinate)
        x = float(new_coordinate[0][0])
        y = float(new_coordinate[1][0])
        z = float(new_coordinate[2][0])
        #print("========================================================")
        #print(previous_coordinate)
        #print(new_coordinate)
        
        return x, y, z 

    
    def _apply_displacement_for_floor(self, x, y, z, dx, dy, dz):
        x += dx
        y += dy
        #z += dz
        return x, y, z


    def _calculate_normal_vector(self):
        index = 0
        self.surface_normals = np.matrix([[0.0, 0.0, 1.0]])
        for point in self.deformed_points:
            if index >0:
                normal = point - self.deformed_points[index-1]
                #normal = point
                normal = normal/np.linalg.norm(normal)
                print(normal)
                self.surface_normals = np.append(self.surface_normals, normal, 0)
            index += 1
        self._calculate_rotation_angle()
        self._calculate_rotation_axis()
        #print(self.surface_normals)


    def _calculate_rotation_axis(self):
        self.rotation_axis = []
        for normal in self.surface_normals:
            #axis = np.cross(normal,[0.0, 0.0, 1.0])
            axis = [0.0, -1.0, 0.0]
            print("rotation axis: ", axis)


    def _calculate_rotation_angle(self):
        initial_vector = np.matrix([[0.0], [0.0], [1.0]])
        self.rotation_angles = []
        for normal in self.surface_normals:
            angle = np.dot(normal,initial_vector)/np.linalg.norm(normal)/np.linalg.norm(initial_vector) 
            angle = np.arccos(np.clip(angle, -1, 1))
            self.rotation_angles.append(angle)
        self.rotation_angles = np.reshape(self.rotation_angles, len(self.rotation_angles))
        print("rotation angle: ", self.rotation_angles * 180/3.14159)

    def plot_1d_structure(self, t):
        """
        plotting the 1d deformation (rotation around x, y axis), torsion cannot be visualized in line structure
        """
        global fig, ax
        x, y, z  = [], [], []
        self.deformed_points   = np.matrix([[0.0, 0.0, 0.0]])
        self.undeformed_points = np.matrix([[0.0, 0.0, 0.0]])
        
        for i in range(0, self.num_of_points_in_height):
            _z = i * self.height / ( self.num_of_points_in_height)
            self.undeformed_points = np.append(self.undeformed_points, np.matrix([[0.0, 0.0, _z]]), 0)
            floor_displacement_x = np.interp( _z, self.undeformed_z, self.vec_dx)
            floor_displacement_y = np.interp( _z, self.undeformed_z, self.vec_dy)
            floor_phi   = np.interp( _z, self.undeformed_z, self.vec_phi)
            floor_beta  = np.interp( _z, self.undeformed_z, self.vec_beta)
            floor_alpha = np.interp( _z, self.undeformed_z, self.vec_alpha)

            _x, _y, _z = self._apply_rotation_for_line_structure(floor_displacement_x, floor_displacement_y, _z, floor_phi, floor_beta, floor_alpha)
            
            x.append(_x)
            y.append(_y)
            z.append(_z)
            self.deformed_points = np.append(self.deformed_points, np.matrix([[_x, _y, _z]]), 0)
        
        self.deformed_points = np.delete(self.deformed_points, 0, 0)
        self.undeformed_points = np.delete(self.undeformed_points, 0, 0)
        self.displacement_vector = self.deformed_points - self.undeformed_points
        self._calculate_normal_vector()
        z = np.array([z, z])
        ax.plot_wireframe(x, y, z, color='bk')


    def plot_frame(self):
        global fig, ax

        frames = [[] for i in range(len(self.floor_geometry["points"]))]
        for floor in self.floors:
            frame = []
            for i in range(0, len(floor)):
                frames[i].append(floor[i])

        for frame in frames:
            x, y, z  = [], [], []
            for i in range(0, len(frame)):
                x.append(frame[i].x)
                y.append(frame[i].y)
                z.append(frame[i].z)

            z = np.array([z, z])
            ax.plot_wireframe(x, y, z)


    def plot_all_floors(self):
        global fig, ax

        for floor in self.floors:
            x, y, z  = [], [], []
            for node in floor:
                x.append(node.x)
                y.append(node.y)
                z.append(node.z)
            x.append(floor[0].x)
            y.append(floor[0].y)
            z.append(floor[0].z)

            z = np.array([z, z])
            ax.set_xlim(-10,10)
            ax.set_ylim(-10,10)
            ax.view_init(10, 45)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.plot_wireframe(x, y, z)


    def update(self, t):
        """
        ploting the deformed structure with respect to the displacement
        """
        global fig, ax

        wframe = fig.gca()
        if wframe!=None:
            ax.cla()

        self._create_all_floors(t)
        self.plot_all_floors()
        self.plot_frame()


    def animate(self):
        # Set up formatting for the movie files
        #Writer = animation.writers['ffmpeg']
        #writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        a = animation.FuncAnimation(fig, self.update, 40, repeat = False)
        #a.save("structure_displacement.avi")
        plt.show()


class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


if __name__ == "__main__":
    floor_geometry = {
        "points": [{
            "x": 5.0,
            "y": 5.0
        },{
            "x": 5.0,
            "y": 0.0
        },{
            "x": 10.0,
            "y": 0.0
        },{
            "x": 10.0,
            "y": -5.0
        },{
            "x": -10.0,
            "y": -5.0
        },{
            "x": -10.0,
            "y": 0.0
        },{
            "x": -5.0,
            "y": 0.0
        },{
            "x": -5.0,
            "y": 5.0
        }]
    }

    floor_geometry_square = {
        "points": [{
            "x": 5.0,
            "y": 5.0
        },{
            "x": 5.0,
            "y": -5.0
        },{
            "x": -5.0,
            "y": -5.0
        },{
            "x": -5.0,
            "y": 5.0
        }]
    }

    num_of_points_in_height = 10
    num_of_floors = 1
    floor_height = 10.0 

    displacement = {
        "dx": lambda t, z: z, #sin(0.05*z)*(2*cos(t)+sin(t)),
        "dy": lambda t, z: 0.0 #2*sin(0.05*z)*(4*cos(t)+3*sin(t))
    }
    rotation = {
        "phi" : lambda t, z: 0, #0.01*z,
        "beta" : lambda t, z: 0, #0.01*z,
        "alpha" : lambda t, z: 0.0 #0.2*sin(0.05*z)*(2*cos(t)+sin(t))
    }
    displacement_pure_torsion = {
        "dx": lambda t, z: t/(t+1e-32) * z/(z+1e-32),
        "dy": lambda t, z: t/(t+1e-32) * z/(z+1e-32)
    }
    myStructure = Structure(displacement, rotation, num_of_points_in_height, num_of_floors, floor_geometry_square, floor_height)
    myStructure.animate()
    myStructure._reconstruct_function_in_vectors(0)
    myStructure._create_all_floors(0)
    myStructure.plot_all_floors()
    myStructure.plot_frame()
    plt.show()