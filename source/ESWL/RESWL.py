import numpy as np
from scipy import signal 
from scipy.signal.windows import hann, boxcar

from source.ESWL.ESWL import ESWL
import source.auxiliary.global_definitions as GD
import source.ESWL.eswl_auxiliaries as auxiliary

class RESWL(object):

    def __init__(self, structure_model, influence_function, load_signals, response, eigenform,  options):

        self.structure_model = structure_model
        self.influence_function_dict = influence_function
        self.options = options
        # the same just in another format for matrix multiplications
        self.load_signals = load_signals
        self.response = response        
        self.load_directions = GD.LOAD_DIRECTION_MAP['all'] 

        self.modes_to_consider = 3
        self.damping_ratio = structure_model.parameters['zeta']
        # to be filled and used outside or internally multiple times
        self.eigenform_unsorted = eigenform
        print('\ncheck and flip signs of mode shapes:')
        for col in range(self.eigenform_unsorted.shape[1]):
            auxiliary.check_and_flip_sign(self.eigenform_unsorted[:,col], col)

        # collect all eigenfroms sorted by label and id of mode 
        self.eigenform_sorted = auxiliary.sort_row_vectors_dof_wise(self.eigenform_unsorted)

        # calculations of things needed multiple times
        self.get_nodal_mass_distribution()
        self.radi_of_gyration = auxiliary.get_radi_of_gyration(self.structure_model)
        self.power_spectral_density_jth_generalized_force()
        self.get_generalized_displacement()
        
        # actual RESWL 
        if self.options['base_moment_distr']:
            self.get_spatial_distribution()
        if self.options['modal_inertial']:
            self.modal_inertial_load()
        self.get_weighting_factors()

    def get_nodal_mass_distribution(self):
        '''
        dictionary with: 'translational': m(z) and 'rotary':I(z) 
        m: equivalent mass at node taken as average of 2 elements below and above node -> for translational dofs
        I_ii: m * (dim_j²+dim_k²)/12;  i, j, k element of [x,y,z] -> for rotational dofs
        NEW: gives a matrix
            rows: nodes
            columns: dof belonging to this mass
        '''
        self.mass_moment_matrix = np.zeros((self.structure_model.n_nodes, GD.DOFS_PER_NODE['3D']))
        storey_x_dim = [self.structure_model.parameters['lx']/self.structure_model.n_elements] * self.structure_model.n_nodes #height

        for n in [0,-1]:
            storey_x_dim[n] = storey_x_dim[n]/2
        self.storey_dims = [storey_x_dim, self.structure_model.parameters['ly'], self.structure_model.parameters['lz']]

        for node_id in range(self.structure_model.n_nodes):
            self.mass_moment_matrix[node_id] = np.asarray([self.structure_model.parameters['m'][node_id]] * GD.DOFS_PER_NODE['3D'])

            for rot in range(3,6):
                d = [0,1,2]
                d.remove(rot-3)
                dim1 = self.storey_dims[d[0]][node_id]
                dim2 = self.storey_dims[d[1]][node_id]
                # matrix that has the diagonals of the actual lumped matrix in a row for each node
                self.mass_moment_matrix[node_id][rot] *= (dim1**2 + dim2**2) / 12

# VARIANT 1: MODAL INERTIAL LOAD

    def modal_inertial_load(self):
        '''
        Variant of computation of resonant component
        eq. 27 Kareem g_r factor is added in ESWL class
        '''
        
        self.modal_inertial = {self.response: {}}# using a lumped mass vector -> see get_nodal_mass
        
        for direction in self.load_directions:
            self.modal_inertial[self.response][direction] = []

        # This is a lumped formulation as it comes from the literature
        for d_i, direction in enumerate(self.load_directions):
            for mode_id in range(self.modes_to_consider):
                generalized_displacement = self.generalized_displacements[mode_id]
                m_z = self.mass_moment_matrix[:,d_i]
                phi_i =  self.eigenform_sorted[direction][:,mode_id]
                D_j = (2*np.pi * self.structure_model.eig_freqs[mode_id])**2
                result = D_j * m_z * phi_i * generalized_displacement

                self.modal_inertial[self.response][direction].append(result)
      
# VARIANT 2: DISTRIBUTION OF RESONANT BASE MOMENT

    def get_spatial_distribution(self):
        '''
        this distributes a resonant base moment along the height
        the spatial distribution for each mode is collected in a list
        each entry of the list is a dictionary with the differetn load directions
        NOTE: never explicity describe for nodal bending moments -> used the formulation as it is described for the torque (Kareem eq. 30 e.g.)
        '''
        self.spatial_distribution = {self.response: {}}
        load_moment_response_map = {'y':'Mz', 'z':'My', 'a': 'Mx', 'b':'My', 'g':'Mz'}
        for d_i, direction in enumerate(self.load_directions):

            self.spatial_distribution[self.response][direction] = []
    
            response_type = load_moment_response_map[direction]
            
            for mode_id in range(self.modes_to_consider):
                # TODO: here nothing is consisten sofar -> guess not really possible
                # NOTE: probably this is actually not so correct if coupled motion is used? see derivation of resonant part e.g. Boggs eq. 14+
                jth_resonant_modal_base_moment = abs(self.get_resonant_modal_base_moment(mode_id, response_type))

                # d_i+1 to leave out x
                numerator = np.multiply(self.mass_moment_matrix[:,d_i+1],  self.eigenform_sorted[direction][:,mode_id]) *\
                            jth_resonant_modal_base_moment
                
                denominator = 0.0
                
                for node, z in enumerate(self.structure_model.nodal_coordinates["x0"]):
                    if direction in ['a','b','g']:
                        denominator += self.mass_moment_matrix[node][d_i+1] * self.eigenform_sorted[direction][:,mode_id][node]
                    else:# multiplied with z
                        denominator += self.mass_moment_matrix[node][d_i+1] * self.eigenform_sorted[direction][:,mode_id][node] * z
                
                # variant with integration
                if direction in ['a','b','g']:
                    # +1 to skip x direction 
                    integrand = np.multiply(self.mass_moment_matrix[:,d_i+1], self.eigenform_sorted[direction][:,mode_id])
                else:
                    integrand = np.multiply(np.multiply(self.mass_moment_matrix[:,d_i+1], self.eigenform_sorted[direction][:,mode_id]),
                                        self.structure_model.nodal_coordinates["x0"]) 

                denominator_int = auxiliary.integrate_influence(integrand, self.structure_model.nodal_coordinates["x0"])
                
                if direction in ['a','b','g']:
                    # NOTE this is wrong cant be replaced 
                    rad_factor = self.radi_of_gyration[direction][1] / sum(self.radi_of_gyration[direction])
                    # if direction == 'a':
                    #     rad_factor = 1 / self.structure_model.n_nodes#self.structure_model.nodal_coordinates['x0'][1]/ (self.structure_model.nodal_coordinates['x0'][-1])
                    # else:
                    #     rad_factor = 1 / self.structure_model.n_nodes
                    denom = denominator# * self.structure_model.nodal_coordinates['x0'][-1]#_int 
                else:
                    rad_factor = 1
                    denom = denominator

                result = numerator / denominator #* rad_factor

                # NOTE result should be the base moment distirbuted along the height

                inf = self.influence_function_dict[self.response][direction]
                base_m = sum(inf * result)

                self.spatial_distribution[self.response][direction].append(result)

    def get_resonant_modal_base_moment(self, mode_id, response_type):
        '''
        retrun the resonant modal base moment of mode i for base moment R
        Kareem eq. 29/30
        '''
        
        participation_coeff = self.get_participation_coefficient(mode_id, response_type)

        return participation_coeff * self.generalized_displacements[mode_id]

    def get_participation_coefficient(self, mode_id, response):
        '''
        participation coefficient of the jth mode to the response r
        Kareem eq. 16
        '''
        # TODO all this could be saved in a dictionary.. it is called multiple times with the same arguments

        self.participation_coeffs_of_jth_mode = []
        
        # lumped formulation
        T_j_Ms = 0.0
        for d_i, direction in enumerate(self.load_directions):
            #inf = self.influence_function_dict[response][direction]
            T_j_Ms += sum(np.multiply(np.multiply(self.influence_function_dict[response][direction], self.mass_moment_matrix[:,d_i+1]),
                    self.eigenform_sorted[direction][:,mode_id]))

        T_j_Ms *= (2*np.pi * self.structure_model.eig_freqs[mode_id])**2
    
        #self.participation_coeffs_of_jth_mode.append(T_j_Ms)
        return T_j_Ms

    def get_generalized_displacement(self):
        '''
        rms resonant component of the jth generalized displacement 
        Kareem eq. 8
        '''
        self.generalized_displacements = np.zeros(self.modes_to_consider)
        for mode_id in range(self.modes_to_consider):
        
            S_Q_jj = self.psd_of_jth_generalized_force[mode_id]
            f_j = self.structure_model.eig_freqs[mode_id]
            n_j = self.damping_ratio
        
            # LUMPED
            M_j = 0.0 
            for node in range(self.structure_model.n_nodes):
                for d_i, direction in enumerate(GD.DOF_LABELS['3D']):
                    M_j_i = self.mass_moment_matrix[node][d_i] * \
                        self.eigenform_sorted[direction][:,mode_id][node]**2
                    M_j += M_j_i

            sig_q_j_r =  np.pi * f_j * S_Q_jj / (M_j**2 * (2*np.pi*f_j)**4 * 4*n_j)    

            # rms value of generalized displacement
            # returning the sqrt 
            self.generalized_displacements[mode_id] = np.sqrt(sig_q_j_r)

    def power_spectral_density_jth_generalized_force(self, window_type = 'box'):
        '''
        the power spectral density of the generalized force of the jth eigenmode
        -> is directly evaluated at the first 3 natural frequencies 
        Kareem eq. 6
        it seems that using window_type boxcar is the best choice, -> varified by using int_{CSD(x,x)} = std(x)
        see CSD_params.py
        ''' 

        self.psd_of_jth_generalized_force = []
        S_Q_jj_i = 0 # jj for ith mode 

        plot_csd = False
        
        f_sample = self.load_signals['sample_freq'] 
        for mode_id in range(self.modes_to_consider):
            for s in self.load_directions:
                for l in self.load_directions:
                    for i1 in range(self.structure_model.n_nodes):
                        for i2 in range(self.structure_model.n_nodes):
                
                            win = boxcar(len(self.load_signals[s][i1]),False)
                            
                            f_csd, csd = signal.csd(self.load_signals[s][i1], self.load_signals[l][i2], f_sample, window= win)

                            # find the CSD at f_j -> needs interpolation
                            f_j = self.structure_model.eig_freqs[mode_id]
                            f_round = np.round(f_csd,1)
                            
                            f_id = np.where(f_round == np.round(f_j,1))[0] #actually this is not necesary but sensible to shorten things
                            
                            # find the f_i that is the closest to f_j
                            if len(f_id) > 1:
                                possible_fs = f_csd[f_id[0]:f_id[-1]+1] 
                                difs = abs(possible_fs - f_j)
                                use_id = int(np.argmin(difs))
                                f_id_closest = f_id[use_id]
                            else:
                                f_id_closest = f_id[0]
                            
                            # find interpolation frequency
                            if f_j < f_csd[f_id_closest]:
                                xp = [f_csd[f_id_closest - 1], f_csd[f_id_closest]]
                                yp = [csd[f_id_closest - 1], csd[f_id_closest]]
                            elif f_j > f_csd[f_id_closest]:
                                xp = [f_csd[f_id_closest], f_csd[f_id_closest+1]]
                                yp = [csd[f_id_closest], csd[f_id_closest+1]]

                            # interpolate the csd value for f_j between two freqs
                            csd_f_j = np.interp(f_j, xp, yp)

                            # NOTE: taking the real part of the csd (e.g. Holmes 3.3.6)
                            S_Q_jj_i += self.eigenform_sorted[s][:,mode_id][i1]*\
                                        self.eigenform_sorted[l][:,mode_id][i2]*\
                                        np.real(csd_f_j)

            # for each of the first 3 modes 
            self.psd_of_jth_generalized_force.append(S_Q_jj_i)

# FUNCTIONS REQUIRED FOR BOTH VARIANTS

    def get_weighting_factors(self):
        '''
        computes the weighting_factors for the first 3 modes
        raw it is just std of the resonant response 
        Kareem eq. 34
        '''
        
        self.weighting_factors_raw = {self.response: {}}

        for direction in self.load_directions:
            self.weighting_factors_raw[self.response][direction] = []
            for mode_id in range(self.modes_to_consider):
                generalized_displacement = self.generalized_displacements[mode_id] # this is already the sqrt 
                participation_coeff = self.get_participation_coefficient(mode_id, self.response)

                self.weighting_factors_raw[self.response][direction].append(participation_coeff * generalized_displacement)

    def get_rms_resonant_response(self, mode_id):
        '''
        rms resonant response for a specific mode -> to be summed up
        intermodal coupling is negelcted (for CAARC not relevant anyways)
        Kareem eq. 17
        '''
        participation_coeff = self.get_participation_coefficient(mode_id, self.response)
        rms_resonant_response = participation_coeff**2 * self.generalized_displacements[mode_id]**2
        return rms_resonant_response

