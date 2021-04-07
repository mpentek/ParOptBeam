import numpy as np
from scipy import signal 
import matplotlib.pyplot as plt 
from scipy.signal.windows import hann, boxcar

from source.ESWL.ESWL import ESWL
import source.auxiliary.global_definitions as GD
from source.ESWL.eswl_auxiliaries import check_and_flip_sign
from source.ESWL.eswl_plotters import plot_n_mode_shapes 

response_displacement_map = {'Qx': 'x', 'Qy': 'y', 'My': 'z', 'Mz': 'y', 'Mx': 'a'}

class RESWL(object):

    def __init__(self, structure_model, influence_function, eigenvalue_analysis, load_signals, load_directions, response, include_all_rotations, lumped):

        self.structure_model = structure_model
        self.influence_function_dict = influence_function
        # the same just in another format for matrix multiplications
        self.influence_vectors = self.prepare_for_consistent_calculations()
        self.eigenvalue_analysis = eigenvalue_analysis
        self.load_signals = load_signals
        self.response = response        
        self.load_directions = load_directions # TODO: find which directions are available and sensible
        self.include_all_rotations = include_all_rotations
        self.lumped = lumped
        if self.include_all_rotations:
            self.load_directions = ['y','z','a','b','g']
            print ('\nusing all load directions', self.load_directions)
            
        else:
            print('\nusing only:', self.load_directions)

        self.T_j_Ms_track = {}

        self.modes_to_consider = 3
        self.damping_ratio = 0.01 # TODO: find out how to compute or take it from the paper -> should be ok like this

        # to be filled and used outside or internally multiple times
        self.eigenform_unsorted = eigenvalue_analysis.eigenform
        print('\ncheck and flip signs of mode shapes:\n')
        for col in range(self.eigenform_unsorted.shape[1]):
            check_and_flip_sign(self.eigenform_unsorted[:,col], col)

        # collect all eigenfroms sorted by label and id of mode 
        self.eigenform_sorted = self.sort_row_vectors_dof_wise(self.eigenform_unsorted)

        #plot_n_mode_shapes(self.eigenform_sorted, self.structure_model.charact_length)
        
        # calculations of things needed 
        self.get_nodal_mass_distribution()
        #self.get_participation_coefficient(self.response)
        self.power_spectral_density_jth_generalized_force()

        # mode specific 
        #self.participation_coeffs_of_jth_mode = np.zeros(self.modes_to_consider)
        self.generalized_displacements = np.zeros(self.modes_to_consider)
        self.rms_resonant_response = 0.0

        for mode_id in range(self.modes_to_consider):
            self.generalized_displacements[mode_id] = self.get_generalized_displacement(mode_id)
            participation_coeff = self.get_participation_coefficient(mode_id, self.response)
            self.rms_resonant_response += participation_coeff**2 * self.generalized_displacements[mode_id]**2

         
        self.get_spatial_distribution()
        self.modal_inertial_load()
        self.get_weighting_factors()

    def modal_inertial_load(self):
        '''
        eq. 27 Kareem g_r is added in ESWL class
        '''
        
        self.modal_inertial = {self.response: {}}
        self.modal_inertial_lumped = {self.response: {}}
        #if self.lumped:
        for d_i, direction in enumerate(self.load_directions):

            self.modal_inertial_lumped[self.response][direction] = []
            self.modal_inertial[self.response][direction] = []
            for mode_id in range(self.modes_to_consider):
                generalized_displacement = self.generalized_displacements[mode_id]
                m_z = self.mass_moment_matrix[:,d_i]
                phi_i =  self.eigenform_sorted[direction][:,mode_id]
                D_j = (2*np.pi * self.structure_model.eig_freqs[mode_id])**2
                result = D_j * m_z * phi_i * generalized_displacement

                self.modal_inertial_lumped[self.response][direction].append(result)

        #else: 
        for mode_id in range(self.modes_to_consider):
            generalized_displacement = self.generalized_displacements[mode_id]
            phi_i =  self.eigenform_unsorted[:,mode_id]
            phi_i_res = phi_i.reshape(phi_i.shape[0], 1)
            D_j = (2*np.pi * self.structure_model.eig_freqs[mode_id])**2
            m = self.structure_model.m
            result = D_j * generalized_displacement * np.matmul(phi_i, m)
            result_right = D_j * generalized_displacement * np.matmul(m, phi_i)
            result_sorted = self.sort_row_vectors_dof_wise(result)
            for d_i, direction in enumerate(self.load_directions):
                self.modal_inertial[self.response][direction].append(result_sorted[direction])
        

    def get_spatial_distribution(self):
        '''
        this distributes a resonant base moment along the height
        the spatial distribution for each mode is collected in a list
        each entry of the list is a dictionary with the differetn load directions
        '''
        self.spatial_distribution = {self.response: {}}
        load_moment_response_map = {'y':'Mz', 'z':'My', 'a': 'Mx', 'b':'My', 'g':'Mz'}
        for d_i, direction in enumerate(self.load_directions):

            self.spatial_distribution[self.response][direction] = []

            response_type = load_moment_response_map[direction]
            
            for mode_id in range(self.modes_to_consider):
                # TODO: here nothing is consisten sofar
                # is this actually not so correct if coupled motion is used? see derivation of resonant part e.g. Boggs eq. 14+
                jth_resonant_modal_base_moment = self.get_resonant_modal_base_moment(mode_id, response_type)

                numerator = np.multiply(self.mass_moment_matrix[:,d_i+1],  self.eigenform_sorted[direction][:,mode_id]) * jth_resonant_modal_base_moment
                #denominator = np.zeros(self.structure_model.n_nodes)
                denominator = 0.0

                for node, z in enumerate(self.structure_model.nodal_coordinates["x0"]):
                    if direction in ['a','b','g']:
                        denominator += self.mass_moment_matrix[node][d_i+1] * self.eigenform_sorted[direction][:,mode_id][node]
                    else:# multiplied with z
                        denominator += self.mass_moment_matrix[node][d_i+1] * self.eigenform_sorted[direction][:,mode_id][node] * z
                
                result = numerator / denominator

                self.spatial_distribution[self.response][direction].append(result)

    def get_nodal_mass_distribution(self):
        '''
        dictionary with: 'translational': m(z) and 'rotary':I(z) 
        m: equivalent mass at node taken as average of 2 elements below and above node -> for translational dofs
        I_ii: m * (dim_j²+dim_k²)/12 i, j, k element of [x,y,z] -> for rotational dofs
        NEW: gives a matrix
            rows: nodes
            columns: dof belonging to this mass
        '''
        self.nodal_mass_distribution = {} #nodal mass distriution and mass moment matrix is the same, the first just for y,z,a
        self.mass_moment_matrix = np.zeros((self.structure_model.n_nodes, GD.DOFS_PER_NODE['3D']))
        storey_x_dim = [self.structure_model.parameters['lx']/self.structure_model.n_elements] * self.structure_model.n_nodes #height

        # VARIANTE 1
        for n in [0,-1]:
            #storey_mass[n] = self.structure_model.parameters['m'][n] / 2
            storey_x_dim[n] = storey_x_dim[n]/2
        storey_dims = [storey_x_dim, self.structure_model.parameters['ly'], self.structure_model.parameters['lz']]


        for node_id in range(self.structure_model.n_nodes):
            self.mass_moment_matrix[node_id] = np.asarray([self.structure_model.parameters['m'][node_id]] * GD.DOFS_PER_NODE['3D'])

            for rot in range(3,6):
                d = [0,1,2]
                d.remove(rot-3)
                dim1 = storey_dims[d[0]][node_id]
                dim2 = storey_dims[d[1]][node_id]
                # matrix that has the diagonals of the actual lumped matrix in a row for each node
                self.mass_moment_matrix[node_id][rot] *= (dim1**2 + dim2**2) / 12


    def get_resonant_modal_base_moment(self, mode_id, response_type):
        '''
        retrun the resonant modal base moment of mode i for base moment r
        '''
        
        participation_coeff = self.get_participation_coefficient(mode_id, response_type)
        # TODO: could make the displacement an attribute since it is only depent on the mode_id and is called and calcualted new for each direction
        generalized_displacement = self.get_generalized_displacement(mode_id)

        return participation_coeff * generalized_displacement

    def get_participation_coefficient(self, mode_id, response):
        '''
        participation coefficient of the jth mode to the response r
        '''
        # TODO all this could be saved in a dict.. it is called multiple times with the same arguments

        self.participation_coeffs_of_jth_mode = []
        self.inf_m = []
        #for mode_id in range(self.modes_to_consider):
        
        # NOTE this is a lumped formulation
        if self.lumped:
            T_j_Ms = 0.0
            for d_i, direction in enumerate(self.load_directions):
                for node in range(self.structure_model.n_nodes):
                    # d_i +1 in mass moment matrix to skip x direction
                    T_j_Ms += ( self.influence_function_dict[response][direction][node] * \
                                self.mass_moment_matrix[node][d_i+1] * \
                                self.eigenform_sorted[direction][:,mode_id][node])

            T_j_Ms *= (2*np.pi * self.structure_model.eig_freqs[mode_id])**2

            self.T_j_Ms_track[mode_id] = T_j_Ms
        
            #self.participation_coeffs_of_jth_mode.append(T_j_Ms)
            return T_j_Ms

        else:
            m = self.structure_model.m
            
            self.inf_m.append(np.matmul(self.influence_vectors[response], m))
            # is this really a consistent mass formulation?
            T_j_ms = np.matmul(np.matmul(self.influence_vectors[response], m), 
                                self.eigenform_unsorted[:,mode_id])[0] *\
                            (2*np.pi * self.structure_model.eig_freqs[mode_id])**2

            #self.participation_coeffs_of_jth_mode.append(T_j_ms)
            return T_j_ms 

    def get_generalized_displacement(self, mode_id):
        '''
        rms resonant component of the jth generalized displacement 
        '''
        S_Q_jj = self.psd_of_jth_generalized_force[mode_id]
        f_j = self.structure_model.eig_freqs[mode_id]
        n_j = self.damping_ratio
        M_j = 0.0 # TODO: if mass normalized?! -> for y and z it should get 1.0 for a which is realted to I not
        phi_i = self.eigenform_unsorted[:,mode_id]
        #if self.lumped:
        #M_j_matmul = np.matmul(np.matmul(np.transpose(phi_i) * self.mass_moment_matrix), phi_i) 
        for node in range(self.structure_model.n_nodes):
            for d_i, direction in enumerate(GD.DOF_LABELS['3D']):
                M_j_i = self.mass_moment_matrix[node][d_i] * \
                    self.eigenform_sorted[direction][:,mode_id][node]**2
                if direction in ['a','b','g']:
                    I = self.mass_moment_matrix[node][d_i]
                    m_i = self.mass_moment_matrix[node][GD.DOF_LABELS['3D'].index('y')] # any translational dof has the storey mass
                    r = I / m_i 
                    M_j_i /= r
                M_j += M_j_i

        result0 =  np.pi * f_j * S_Q_jj / (M_j**2 * (2*np.pi*f_j)**4 * 4*n_j)    
        #else:
        m = self.structure_model.m
        m_j = np.matmul( np.matmul( np.transpose(phi_i) , m ), phi_i)
        result1 = np.pi * f_j * S_Q_jj / (m_j**2 * (2*np.pi*f_j)**4 * 4*n_j)
        # rms value of generalized displacement 
        if self.lumped:
            m_gen = M_j
            result = result0
        else:
            m_gen = m_j
            result = result1

        sig_q_j_r = result

        return np.sqrt(sig_q_j_r)
        
    def power_spectral_density_jth_generalized_force(self, window_type = 'hann'):
        '''
        the power spectral density of the generalized force of the jth eigenmode
        -> is directly evaluated at the first 3 natural frequencies 
        ''' 

        self.psd_of_jth_generalized_force = []
        S_Q_jj_i = 0 # jj for jth mode 

        plot_csd = False
        
        f_sample = self.load_signals['sample_freq'] 
        for mode_id in range(self.modes_to_consider):
            for s in self.load_directions:
                for l in self.load_directions:
                    for i1 in range(self.structure_model.n_nodes):
                        for i2 in range(self.structure_model.n_nodes):
                            if window_type == 'hann':
                                win = hann(len(self.load_signals[s][i1]),False)
                            elif window_type == 'box':
                                win = boxcar(len(self.load_signals[s][i1]),False)
                            
                            if not window_type:
                                f, csd = signal.csd(self.load_signals[s][i1], self.load_signals[l][i2], f_sample, nperseg= 2048)#, window= win)
                            else:
                                f, csd = signal.csd(self.load_signals[s][i1], self.load_signals[l][i2], f_sample, nperseg= 2048, window= win)
                            
                            f_j = self.structure_model.eig_freqs[mode_id]
                            f_round = np.round(f,2)
                            
                            f_id = np.where(f_round ==np.round(f_j,2))[0]
                            # NOTE: sofar: if more then one rounded f exists then the one that is the closest to f_j is taken
                            # TODO:  maybe find the two closest freqs and then interoplate the csd (np.interpl1d, cubic,...)
                            if len(f_id) > 1:
                                possible_fs = f[f_id[0]:f_id[-1]+1] 
                                difs = abs(possible_fs - f_j)
                                use_id = np.argmin(difs)
                                f_id_closest = f_id[use_id]
                            else:
                                f_id_closest = f_id[0]
                            
                            # find interpolation freq
                            if f_j < f[f_id_closest]:
                                xp = [f[f_id_closest - 1], f[f_id_closest]]
                                yp = [csd[f_id_closest - 1], csd[f_id_closest]]
                            elif f_j > f[f_id_closest]:
                                xp = [f[f_id_closest], f[f_id_closest+1]]
                                yp = [csd[f_id_closest], csd[f_id_closest+1]]

                            csd_f_j = np.interp(f_j, xp, yp)

                            real = ' only real'
                            csd_f_j_close = csd[f_id_closest]  
                            if csd_f_j.imag != 0:
                                #NOTE: this is the phase shift and seems not to be relevant 
                                #print('\nCross power spectral density has imaginary part')
                                real = ' with imaginary part'

                            # # plot of the CPSD
                            
                            if plot_csd:
                                plt.semilogy(f, np.real(csd), label = 'csd at f_j: ' + str(csd_f_j))
                                plt.vlines(f[f_id_closest], 0, max(csd), label = 'estimated f', linestyles='-.', color = 'g')
                                plt.vlines(f_j, 0, max(csd), label = 'natural f', linestyles='--', color = 'r')
                                plt.xlabel('frequency')
                                plt.ylabel('CSD ' + 'P_' + s +'_'+ str(i1)+ ' & ' + 'P_' + l +'_'+ str(i2) )
                                plt.title('in mode '+ str(mode_id+1) + ' CSD ' + 'P_' + s +'_'+ str(i1)+ ' & ' + 'P_' + l +'_'+ str(i2) + real)
                                plt.legend()
                                plt.show()

                            # NOTE: taking the real part of the csd (Holmes 3.3.6)
                            S_Q_jj_i += self.eigenform_sorted[s][:,mode_id][i1]*\
                                        self.eigenform_sorted[l][:,mode_id][i2]*\
                                        np.real(csd_f_j)
            
            # for each of the first 3 modes 
            self.psd_of_jth_generalized_force.append(S_Q_jj_i)

    def get_weighting_factors(self):
        '''
        computes the weighting_factors for the first 3 modes
        '''
        
        self.weighting_factors_raw = {self.response: {}}

        for direction in self.load_directions:
            self.weighting_factors_raw[self.response][direction] = []
            for mode_id in range(self.modes_to_consider):
                generalized_displacement = self.generalized_displacements[mode_id]
                participation_coeff = self.get_participation_coefficient(mode_id, self.response)

                self.weighting_factors_raw[self.response][direction].append(
                        participation_coeff * generalized_displacement)

    def prepare_for_consistent_calculations(self):
        '''
        converts the eigenform and influence dictionary into listed 1Dmatrices and only keeps the values of the first 3 modes
        
        returns: influences (one row n_nodes*n_dof collumns), eigenforms (column vector opposite as )

        acces: influences[i], phi[i] i is the ith mode
        '''
        
        phi = []
        n_dofs = GD.DOFS_PER_NODE['3D']
        n_nodes = self.structure_model.n_nodes

        influence_vectors = {}
        for response in self.influence_function_dict:

            influence_vectors[response] = np.zeros((1, n_dofs* n_nodes))
            
            for d_i, direction in enumerate(GD.DOF_LABELS['3D']):
                for node in range(self.structure_model.n_nodes):
                    influence_vectors[response][0][d_i+n_dofs*node] = self.influence_function_dict[response][direction][node]           
        
        return influence_vectors

    def sort_row_vectors_dof_wise(self, unsorted_vector):
        '''
        unsorted vector is of dimenosn n_nodes * n_dofs
        sort it in to a dict with dof lables as keys
        '''
        sorted_dict = {}
        #self.eigenform_sorted = {}
        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                                GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = self.eigenform_unsorted.shape[0] + idx - step
            sorted_dict[label] = unsorted_vector[start:stop+1:step]
        
        return sorted_dict
