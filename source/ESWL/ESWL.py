import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import source.auxiliary.global_definitions as GD
import source.ESWL.eswl_plotters as plotter_utilities

response_labels = ['Qx', 'Qy', 'Qz', 'Mx', 'My', 'Mz']

class ESWL(object):

    def __init__(self, structure_model, eigenvalue_analysis, response, load_signals, resonant_load_directions ,
                load_directions, lumped, decoupled_influences, include_all_rotations = True):
        '''
        is initialize for the response given (base reactions atm):
        [Qx, Qy, Qz, Mx, My, Mz]
        '''
        self.structure_model = structure_model
        self.eigenvalue_analysis = eigenvalue_analysis
        self.load_signals = load_signals
        self.load_directions = load_directions
        self.resonant_load_directions = resonant_load_directions
        self.include_all_rotations = include_all_rotations
        self.lumped = lumped
        
        if response in response_labels:
            self.response = response
        else:
            raise Exception(response + ' is not a valid response label, use: ', response_labels)

        # do this at initialization since it is needed multiple times and pass it to R and B
        self.decoupled_influences = decoupled_influences
        if self.decoupled_influences:
            print ('using decoupled influences\n')
        else:
            print ('using influences from static analysis\n')
        self.initialize_influence_functions(self.decoupled_influences)

        self.eswl_total = {} # [response][direction] = load vector

        self.eswl_components = {}


    def calculate_total_ESWL(self):
        '''
        fills a dictionary with the componentes of the ESWL (mean, background, resonant, total ,lrc)
        '''
        from source.ESWL.BESWL import BESWL
        from source.ESWL.RESWL import RESWL

        #########für jede direction wird jetzt ein neues objekt erzeugt und immer wieder initialisiert -> manche variablen sind aber Richtungs unabhängig 
        self.BESWL = BESWL(self.structure_model, self.influences, self.load_signals, self.load_directions, self.response)
        self.RESWL = RESWL(self.structure_model, self.influences, self.eigenvalue_analysis, 
                            self.load_signals, self.resonant_load_directions, self.response, 
                            self.include_all_rotations, self.lumped) # this is response specific 

        # initialized dictionary of total eswl 
        self.eswl_total[self.response] = {}

        self.eswl_components[self.response] = {}

        for direction in self.load_directions:
            self.eswl_total[self.response][direction] = np.zeros(self.structure_model.n_nodes)
            
            mean_load = np.zeros(self.structure_model.n_nodes)
            for i in range(len(mean_load)):
                mean_load[i] = np.mean(self.load_signals[direction][i])
            
            R_max = self._get_maximum_response(self.response)

            # BESWL
            # TODO: check if the multiplication with peak factor hear is correct-> must this be done twice: seems to be correct
            # seems to be correct since g also is in R_max
            g_b = self._get_peak_factor('background')
            w_b = self.BESWL.weighting_factors_raw[self.response][direction] * g_b / R_max
            #print ('\nBESWL weighting factor for direction', direction, 'is', round(w_b,3))
            p_z_b = self.BESWL.spatial_distribution[self.response][direction] * g_b  
            p_z_b_e = abs(w_b * p_z_b) 

            # ALTERNATIVE for BESWL - LRC KASPERSIK
            # 1
            p_z_b_e_lrc = g_b * self.BESWL.get_beswl_LRC(direction)
            # 2
            p_z_b_e_lrc1 = g_b * self.BESWL.get_beswl_LRC_1(direction)

            #plotter_utilities.plot_rho(self.BESWL.rho_collection_all, self.response)
            
            # RESONANT
            # TODO: seperation of modes -> mentioned in Kareem but not used so far (since expected not to be relevant). Rule: fi+1 > 0.9*fi
            w_r = self.RESWL.weighting_factors_raw[self.response][direction] 
            p_z_r = self.RESWL.spatial_distribution[self.response][direction] 
            # modal inertial load
            p_z_rm = self.RESWL.modal_inertial[self.response][direction]
            p_z_rm_lumped = self.RESWL.modal_inertial_lumped[self.response][direction]
            p_z_r_e = np.zeros(self.structure_model.n_nodes)
            p_z_r_em = np.zeros(self.structure_model.n_nodes)
            p_z_r_em_lumped = np.zeros(self.structure_model.n_nodes)
            for mode_id in range(3):
                g_r = self._get_peak_factor('resonant', self.structure_model.eig_freqs[mode_id])
                p_z_r_e += w_r[mode_id]* g_r / R_max * p_z_r[mode_id] * g_r  
                p_z_r_em += w_r[mode_id]* g_r / R_max * p_z_rm[mode_id] * g_r 
                p_z_r_em_lumped += w_r[mode_id]* g_r / R_max * p_z_rm_lumped[mode_id] * g_r 
            
            # TODO maybe do something here manually that adjusts the signs of g and y
            if direction == 'z':
                if p_z_r_e[1] < 0:
                    sign_z = 'n'
                else:
                    sign_z = 'p'
           
            if direction == 'b':
                if p_z_r_e[1] < 0:
                    if sign_z == 'n':
                        p_z_r_e *= -1
                else:
                    if sign_z == 'p':
                        p_z_r_e *= -1
            
            # # TOTAL
            for node in range(self.structure_model.n_nodes):
                # only needed if mean is negative with abs used before
                if mean_load[node] < 0:
                    p_z_b_e[node] *= -1
                    #p_z_r_e[node] *= -1
                    #p_z_r_em[node] *= -1
                

            self.eswl_total[self.response][direction] = mean_load + p_z_b_e + p_z_r_e
            # collect all load components in one dict to easy access them
            self.eswl_components[self.response][direction] = {}
            self.eswl_components[self.response][direction]['mean'] = mean_load
            self.eswl_components[self.response][direction]['background'] = p_z_b_e
            self.eswl_components[self.response][direction]['resonant'] = p_z_r_e
            self.eswl_components[self.response][direction]['resonant_m'] = p_z_r_em
            self.eswl_components[self.response][direction]['resonant_m_lumped'] = p_z_r_em_lumped
            self.eswl_components[self.response][direction]['total'] = self.eswl_total[self.response][direction]
            self.eswl_components[self.response][direction]['lrc'] = p_z_b_e_lrc
            self.eswl_components[self.response][direction]['lrc1'] = p_z_b_e_lrc1
            #print ('\n added ESWL in load direction', direction)

    def _get_peak_factor(self, response_type = None , frequency = None):
        if not response_type:
            return 3.5

        elif response_type == 'background':
            return 3.5

        elif response_type == 'resonant':
            #g_r = np.zeros(3)
            #for freq in frequencies:
            g_r = np.sqrt(2*np.log(600*frequency)) + 0.5772/np.sqrt(2*np.log(600*frequency))
            #print ('resonant peak factor calculated for a period of 10 min\n')
            return g_r

        elif response_type == 'extreme_value':
            pass
            # z.B. extreme value analysis or as Kaspersik

    def _get_maximum_response(self, response):
        
        # background
        g_b = self._get_peak_factor('background') 
        R_b = g_b**2 * self.BESWL.rms_background_response 

        # resonant 
        g_r = self._get_peak_factor('resonant', self.structure_model.eig_freqs[0])
        R_r = g_r**2 * self.RESWL.rms_resonant_response 

        return np.sqrt(R_b + R_r)

    def initialize_influence_functions(self, decoupled_influences):
        
        # teh influences on the base moments are needed in the resonant computations
        required_responses = ['Mx', 'My', 'Mz']
        if self.response not in required_responses:
            required_responses.append(self.response)

        from source.ESWL.eswl_auxiliaries import get_decoupled_influences
        from source.ESWL.eswl_auxiliaries import get_influence

        self.influences = {}
        for response in required_responses:
            self.influences[response] = {}
            for direction in GD.DOF_LABELS['3D']:
                self.influences[response][direction] = np.zeros(self.structure_model.n_nodes)
                for node in range(self.structure_model.n_nodes):
                    if self.decoupled_influences:
                        
                        
                        influence = get_decoupled_influences(self.structure_model, direction, node, response)
                    else:
                        
                        
                        influence = get_influence(self.structure_model, direction, node, response)
                    self.influences[response][direction][node] = influence

    def plot_eswl_directional_components(self, load_directions, response_label):
        
        plotter_utilities.plot_load_components(self.eswl_total, 
                                                self.structure_model.nodal_coordinates,
                                                load_directions,
                                                response_label)

    def plot_eswl_components(self, response_label, load_directions, textstr, components_to_plot = ['all']):

        plotter_utilities.plot_eswl_components(self.eswl_components,
                                                self.structure_model.nodal_coordinates,
                                                load_directions,
                                                response_label,
                                                textstr,
                                                components_to_plot)


    def postprocess(self):
        pass
        #self.plot_load_components()

