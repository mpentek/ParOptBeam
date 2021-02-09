import numpy as np
from scipy import signal 
import matplotlib.pyplot as plt 

from source.ESWL.ESWL import ESWL
import source.auxiliary.global_definitions as GD

response_displacement_map = {'Qx': 'x', 'Qy': 'y', 'My': 'z', 'Mz': 'y', 'Mx': 'a'}

class RESWL(object):

    def __init__(self, structure_model, influence_function, eigenvalue_analysis, load_signals, load_directions, response):

        self.structure_model = structure_model
        self.influence_function = influence_function
        self.eigenvalue_analysis = eigenvalue_analysis
        self.load_signals = load_signals
        self.response = response        
        self.load_directions = load_directions # TODO: find which directions are available and sensible

        self.modes_to_consider = 3
        self.damping_ratio = 0.01 # TODO: find out how to compute or take it from the paper

        # to be filled and used outside or internally multiple times
        self.eigenform = {}
        # collect all eigenfroms sorted by label and id of mode 
        for idx, label in zip(list(range(GD.DOFS_PER_NODE[self.structure_model.domain_size])),
                              GD.DOF_LABELS[self.structure_model.domain_size]):
            start = idx
            step = GD.DOFS_PER_NODE[self.structure_model.domain_size]
            stop = eigenvalue_analysis.eigenform.shape[0] + idx - step
            self.eigenform[label] = eigenvalue_analysis.eigenform[start:stop+1:step]

        # calculations of things needed 
        self.get_modal_mass_distribution()
        self.power_spectral_density_jth_generalized_force()

        # mode specific 
        self.participation_coeffs_of_jth_mode = np.zeros(self.modes_to_consider)
        self.generalized_displacements = np.zeros(self.modes_to_consider)
        self.rms_resonant_response = 0.0
        for mode_id in range(self.modes_to_consider):
            self.participation_coeffs_of_jth_mode[mode_id] = self.get_participation_coefficient(['y','z','a'], mode_id)
            self.generalized_displacements[mode_id] = self.get_generalized_displacement(mode_id)
            self.rms_resonant_response += self.participation_coeffs_of_jth_mode[mode_id]**2 * self.generalized_displacements[mode_id]**2


        self.weighting_factors_raw = {self.response: {}}
        self.spatial_distribution = {self.response: {}} # key response ist quasi unnötig
        self.get_spatial_distribution()
        self.get_weighting_factors()
    
    def get_spatial_distribution(self):
        
        for direction in self.load_directions:
            if direction == 'a':
                mass_type = 'rotary'
            else:
                mass_type = 'translational'
            self.spatial_distribution[self.response][direction] = []
            for mode_id in range(self.modes_to_consider):

                resonant_modal_base_moment = self.get_resonant_modal_base_moment([direction] ,mode_id)

                numerator = np.multiply(self.modal_mass_distribution[mass_type],  self.eigenform[direction][:,mode_id]) * resonant_modal_base_moment

                denominator = np.zeros(self.structure_model.n_nodes)
                for node, z in enumerate(self.structure_model.nodal_coordinates["x0"][1:],1):
                    denominator[node] = self.modal_mass_distribution[mass_type][node] * self.eigenform[direction][:,mode_id][node] * z
                
                result = np.zeros(self.structure_model.n_nodes)
                result[1:] += np.divide(numerator[1:],denominator[1:])

#### hier wird durch 0 geteilt!
                self.spatial_distribution[self.response][direction].append(result)

    def get_modal_mass_distribution(self):
        '''
        dictionary with: 'translational': m(z) and 'rotary':I(z)
        m: equivalent mass at node taken as average of 2 elements below and above node
        I: m * (by²+bz²)/12
        '''
        self.modal_mass_distribution = {}

        
        for direction in ['translational', 'rotary']:
            storey_mass = np.zeros(self.structure_model.n_nodes) #ground node remains 0 mass
            for el_idx in range(self.structure_model.n_elements):

                storey_mass[el_idx+1] = (self.structure_model.parameters['m'][el_idx] + self.structure_model.parameters['m'][el_idx+1])/2

                if direction == 'rotary':
                    # this is not so correct since here only the current element is taken and not the 2 node bounding ones as for the mass
                    storey_mass[el_idx+1] *= (self.structure_model.parameters['lz'][el_idx] **
                                    2 + self.structure_model.parameters['ly'][el_idx] ** 2) / 12
            # last node outside loop
            storey_mass[-1] = self.structure_model.parameters['m'][el_idx] / 2
            if direction == 'rotary':
                storey_mass[-1] *= (self.structure_model.parameters['lz'][-1] **
                                    2 + self.structure_model.parameters['ly'][-1] ** 2) / 12

            # add to dictionary                       
            self.modal_mass_distribution[direction] = storey_mass

    def get_resonant_modal_base_moment(self, directions, mode_id):
        '''
        retrun the resonant modal base moment of mode i 
        '''
        participation_coeff = self.get_participation_coefficient(directions, mode_id)
        generalized_displacement = self.get_generalized_displacement(mode_id)

        return participation_coeff * generalized_displacement

    def get_participation_coefficient(self, directions, mode_id):

        T_j_Ms = 0.0
        for direction in directions:
            if direction == 'a':
                mass_type = 'rotary'
            else:
                mass_type = 'translational'
            for node , z in enumerate(self.structure_model.nodal_coordinates['x0'][1:],1):
                T_j_Ms += self.influence_function[self.response][direction][node] * self.modal_mass_distribution[mass_type][node] * \
                        self.eigenform[direction][:,mode_id][node]
                        
        T_j_Ms *= (2*np.pi * self.structure_model.eig_freqs[mode_id])

        return T_j_Ms

    def get_generalized_displacement(self, mode_id):
        '''
        resonant component of the jth generalized displacement 
        '''
        S_Q_jj = self.psd_of_jth_generalized_force[mode_id]
        f_j = self.structure_model.eig_freqs[mode_id]
        n_j = self.damping_ratio
        M_j = 0.0 # TODO: if mass normalized?!
        for node , z in enumerate(self.structure_model.nodal_coordinates['x0'][1:],1):
            for direction in ['y','z','a']:
                if direction == 'a':
                    mass_type = 'rotary'
                else:
                    mass_type = 'translational'
                M_j += self.modal_mass_distribution[mass_type][node] * \
                    self.eigenform[direction][:,mode_id][node]**2
                

        # rms value of generalized displacement 
        sig_q_j_r = np.pi * f_j * S_Q_jj / (M_j**2 *(2*np.pi*f_j)**2 *4*n_j)

        # TODO: is the square? --> see Gl. 8 Kareem
        return np.sqrt(sig_q_j_r)
        
    def power_spectral_density_jth_generalized_force(self):
        '''
        the power spectral density of the generalized force of the jth eigenmode
        -> is directly evaluated at the first 3 natural frequencies
        '''
        self.psd_of_jth_generalized_force = []
        S_Q_jj_i = 0 # jj for jth mode 

        plot_csd = False

        tol = 0.001
        f_sample = 10.0 # default --> TODO: or 1/dt with dt from simulation 
        for mode_id in range(self.modes_to_consider):
            for s in self.load_directions:
                for l in self.load_directions:
                    for i1, z1 in enumerate(self.structure_model.nodal_coordinates['x0'][1:],1):
                        for i2, z2 in enumerate(self.structure_model.nodal_coordinates['x0'][1:],1):
                            f_j = self.structure_model.eig_freqs[mode_id]
                            f, csd = signal.csd(self.load_signals[s][i1], self.load_signals[l][i2], f_sample, nperseg= 2048)
                            f_round = np.round(f,2)
                            # here maybe find the tow closest freqs and then interoplate the csd
                            f_id = np.where(f_round ==np.round(f_j,2))[0]

                            if len(f_id) > 1:
                                possible_fs = f[f_id[0]:f_id[-1]+1] 
                                difs = abs(possible_fs - f_j)
                                
                                use_id = np.argmin(difs)
                                f_id_select = f_id[use_id]
                            else:
                                f_id_select = f_id[0]
                            
                            real = ' only real'
                            csd_f_j = csd[f_id_select] # the cross power spectral density evaluated at frequency of current mode abs is taken since it seems that imaginary part is 0 
                            if csd_f_j.imag != 0:
                                #raise Exception('Cross power spectral density has imaginary part')
                                #NOTE: this is the phase shift and seems not to be relevant 
                                #print('\nCross power spectral density has imaginary part')
                                real = ' with imaginary part'

                            # # plot of the CPSD
                            # NOTE: taking the real part of the csd (Holmes 3.3.6)
                            if plot_csd:
                                plt.semilogy(f, np.abs(csd), label = 'csd at f_j: ' + str(csd_f_j))
                                plt.vlines(f[f_id_select], 0, max(csd), label = 'estimated f', linestyles='-.', color = 'g')
                                plt.vlines(f_j, 0, max(csd), label = 'natural f', linestyles='--', color = 'r')
                                plt.xlabel('frequency')
                                plt.ylabel('CSD ' + 'P_' + s +'_'+ str(i1)+ ' & ' + 'P_' + l +'_'+ str(i2) )
                                plt.title('in mode '+ str(mode_id+1) + ' CSD ' + 'P_' + s +'_'+ str(i1)+ ' & ' + 'P_' + l +'_'+ str(i2) + real)
                                plt.legend()
                                plt.show()

                            S_Q_jj_i += self.eigenform[s][:,mode_id][i1]*\
                                        self.eigenform[l][:,mode_id][i2]*\
                                        np.real(csd_f_j)
            
            # for each of the first 3 modes 
            self.psd_of_jth_generalized_force.append(S_Q_jj_i)

    def get_weighting_factors(self):
        '''
        computes the weighting_factors for the first 3 modes
        '''
        for direction in self.load_directions:
            self.weighting_factors_raw[self.response][direction] = []
            for mode_id in range(self.modes_to_consider):
                participation_coeff = self.get_participation_coefficient([direction], mode_id)
                generalized_displacement = self.generalized_displacements[mode_id]

                # integrated here because other wise it would be double 
                self.weighting_factors_raw[self.response][direction].append(participation_coeff * generalized_displacement)
