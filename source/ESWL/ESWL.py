import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import timeit

import source.auxiliary.global_definitions as GD
import source.ESWL.eswl_plotters as plotter_utilities
import source.ESWL.eswl_auxiliaries as auxiliary

response_labels = ['Qx', 'Qy', 'Qz', 'Mx', 'My', 'Mz']

class ESWL(object):

    def __init__(self, structure_model, eigenvalue_analysis, response, load_signals, load_directions,
                 use_lumped, decoupled_influences, use_lrc, plot_mode_shapes = False):
        '''
        is initialize for the response given (base reactions atm):
        [Qx, Qy, Qz, Mx, My, Mz]
        '''
        self.structure_model = structure_model
        self.eigenvalue_analysis = eigenvalue_analysis
        self.load_signals = load_signals
        self.load_directions = load_directions
        self.use_lumped = use_lumped 
        self.use_lrc = use_lrc
        self.plot_mode_shapes = plot_mode_shapes
        
        if response in response_labels:
            self.response = response
        else:
            raise Exception(response + ' is not a valid response label, use: ', response_labels)

        # do this at initialization since it is needed multiple times and pass it to R and B
        self.decoupled_influences = decoupled_influences
        if self.decoupled_influences:
            print ('\nusing decoupled influences\n')
        else:
            print ('\nusing influences from static analysis\n')
        self._initialize_influence_functions(self.decoupled_influences)

        self.eswl_components = {} # first key is the response, second the load direciton
        self.eswl_total = {}

    def calculate_total_ESWL(self):
        '''
        fills a dictionary with the componentes of the ESWL (mean, background, resonant, total ,lrc)
        '''
        from source.ESWL.BESWL import BESWL
        from source.ESWL.RESWL import RESWL

        #für jede direction wird jetzt ein neues objekt erzeugt und immer wieder initialisiert -> manche variablen sind aber Richtungs unabhängig 
        self.BESWL = BESWL(self.structure_model, self.influences, self.load_signals, self.load_directions, self.response)
        self.RESWL = RESWL(self.structure_model, self.influences, self.eigenvalue_analysis, 
                            self.load_signals, self.load_directions, self.response, 
                            self.use_lumped, self.plot_mode_shapes) 

        self.eswl_components[self.response] = {}
        self.eswl_total[self.response] = {}

        # maximum response
        R_max = self._get_maximum_response(self.response)

        print ('\nESWL Component Combinations:')
        # calculate the eswl for each load direction
        for direction in self.load_directions:
            
            # =======================================================================
            # # MEAN PART 
            mean_load = np.asarray([np.mean(x) for x in self.load_signals[direction]])
            
            # =======================================================================
            # BESWL - BACKGROUND PART 
            g_b = self._get_peak_factor('background')
            w_b = self.BESWL.weighting_factors_raw[self.response][direction] * g_b / R_max
            p_z_b = self.BESWL.spatial_distribution[self.response][direction] * g_b  
            p_z_b_e = abs(w_b * p_z_b) 

            # # ALTERNATIVE for BESWL - LRC KASPERSIK
            w_b_lrc = np.sqrt(self.BESWL.rms_background_response) * g_b / R_max # weoghting factor for combinations Kareem eq. 33, Holmes eq.4.41
            p_z_b_lrc = g_b * abs(self.BESWL.get_beswl_LRC(direction))
            p_z_b_e_lrc = w_b_lrc * p_z_b_lrc

            if self.use_lrc:
                background = p_z_b_e_lrc
            else:
                background = p_z_b_e

            # # SIGN OF BACKGROUND COMPONENTN
            # background is a weighted std, thus it should increase the mean value absolutley
            for node in range(self.structure_model.n_nodes):
                if mean_load[node] < 0:
                    background[node] *= -1

            # =======================================================================
            # # RESWL - RESONANT PART 
            # NOTE: seperation of modes -> Rule: fi+1*0,9 > fi -> given for CAARC 
            w_r_j = self.RESWL.weighting_factors_raw[self.response][direction] 
            # 1. variante: distribute resonant base moment along height Kareem eq. 29
            p_z_r = self.RESWL.spatial_distribution[self.response][direction] 
            # 2. modal inertial load: Kareem eq. 27
            p_z_rm = self.RESWL.modal_inertial[self.response][direction]
            p_z_rm_lumped = self.RESWL.modal_inertial_lumped[self.response][direction]
            p_z_r_e = np.zeros(self.structure_model.n_nodes)
            p_z_r_em = np.zeros(self.structure_model.n_nodes)
            p_z_r_em_lumped = np.zeros(self.structure_model.n_nodes)

            # sum up over first 3 modes and multiply with weighting factor
            for mode_id in range(3):
                g_r = self._get_peak_factor('resonant', self.structure_model.eig_freqs[mode_id])
                w_r = w_r_j[mode_id] * g_r / R_max # weighting factor

                # multiplying p_z_r_ with g_r according to eq. 27,29 Kareem
                p_z_r_e +=  w_r * p_z_r[mode_id] * g_r  
                p_z_r_em += w_r * p_z_rm[mode_id] * g_r 
                p_z_r_em_lumped += w_r * p_z_rm_lumped[mode_id] * g_r 
            
            # # SIGN OF RESONANT PART
            # 1. select sign du to combination with other components
            resonant_sign, flipped_sign = self.get_sign_for_resonant(mean_load ,background ,p_z_r_e, direction)

            if flipped_sign:
                print ('\n   flipped sign of resonant', direction, ', reason: component combination')
            
            p_z_r_e = resonant_sign*p_z_r_e
            p_z_r_em = resonant_sign*p_z_r_em
            p_z_r_em_lumped = resonant_sign*p_z_r_em_lumped

            '''
            NOTE: here manually adjusting the signs of b(My), g(Mz) coupled with z(Fz), y(Fy) of resonant component -> should be overcome with the mode shape sign flips???
            since resonant loads are dependent on the mode shapes and b,z and g,y are coupled the signs must be coupled aswell
            '''

            if direction == 'y':
                if p_z_r_e[1] < 0:
                    sign_y = 'n'
                else:
                    sign_y = 'p'
           
            if direction == 'g': 
                if p_z_r_e[1] < 0:
                    if sign_y == 'p':
                        print ('  flipped sign of resonant g, reason: coupled y-g')
                        flipped_g_coupling = True
                        p_z_r_e *= -1
                    else:
                        flipped_g_coupling = False
                else:
                    if sign_y == 'n':
                        print ('  flipped sign of resonant g, reason: coupled y-g')
                        flipped_g_coupling = True
                        p_z_r_e *= -1
                    else:
                        flipped_g_coupling = False

            if direction == 'z':
                if p_z_r_e[1] < 0:
                    sign_z = 'n'
                else:
                    sign_z = 'p'
           
            if direction == 'b': 
                if p_z_r_e[1] < 0:
                    if sign_z == 'n':
                        print ('  flipped sign of resonant b, reason: coupled z-b')
                        flipped_b_coupling =True
                        p_z_r_e *= -1
                    else:
                        flipped_b_coupling = False
                else:
                    if sign_z == 'p':
                        print ('  flipped sign of resonant b, reason: coupled z-b')
                        flipped_b_coupling = True
                        p_z_r_e *= -1
                    else:
                        flipped_b_coupling = False
            
            # =======================================================================
            # # TOTAL

            eswl_total = mean_load + background + p_z_r_e

            # this is needed in this format for postprocessing
            self.eswl_total[self.response][direction] = mean_load + background + p_z_r_e

            # collect all load components in one dict to easy access them
            self.eswl_components[self.response][direction] = {}
            self.eswl_components[self.response][direction]['mean'] = mean_load
            self.eswl_components[self.response][direction]['background'] = p_z_b_e
            self.eswl_components[self.response][direction]['lrc'] = p_z_b_e_lrc
            self.eswl_components[self.response][direction]['resonant'] = p_z_r_e
            self.eswl_components[self.response][direction]['resonant_m'] = p_z_r_em
            self.eswl_components[self.response][direction]['resonant_m_lumped'] = p_z_r_em_lumped
            self.eswl_components[self.response][direction]['total'] = eswl_total
            

    def _get_peak_factor(self, response_type = None , frequency = None):
        '''
        for resonant part: caculation according a standard formula
        for background part: so far fixed to 3.5 (could maybe be done differently but mostyl used like this)
        '''
        if not response_type:
            return 3.5

        elif response_type == 'background':
            return 3.5

        elif response_type == 'resonant':
            g_r = np.sqrt(2*np.log(600*frequency)) + 0.5772/np.sqrt(2*np.log(600*frequency))
            return g_r

        elif response_type == 'extreme_value':
            pass
            # z.B. extreme value analysis or as Kaspersik 2009

    def _get_maximum_response(self, response):
        '''
        maximum response according to e.g. Holmes eq. 5.41 and Kareem eq. 18
        '''
        # background
        g_b = self._get_peak_factor('background') 
        R_b = g_b**2 * self.BESWL.rms_background_response 

        # resonant 
        # for each response one mode is the most relevant, the g_R associated with this mode is then the one mostly contributing to the response 
        R_r = 0.0
        for mode_id in range(3):
            sig_R_r = self.RESWL.get_rms_resonant_response(mode_id)
            g_r = self._get_peak_factor('resonant', self.structure_model.eig_freqs[mode_id])
            R_r += g_r**2 * sig_R_r

        return np.sqrt(R_b + R_r)

    def get_sign_for_resonant(self, mean, background, resonant, current_direction):
        '''
        The background component is already added such that it increases the mean part at each node
        The resonant part must either be added or subtracted at a whole, not node wise.
        Thus here changing the sign such that the maximum response is reached
        '''
        starttime_sign_check = timeit.default_timer()
        input_sign,output_sign = 'p','p'
        if resonant[1] < 0:
            input_sign ='n'
        responses = []
        signs = [1.0,-1.0]
        loads = []
        influence = self.influences[self.response][current_direction]
        for sign in signs:
            total_load = mean + background + sign*resonant
            loads.append(total_load)
            R = sum(np.multiply(influence, total_load))

            responses.append(abs(R))

        endtime_sign_check = timeit.default_timer()

        #print ('\nneeded:', round(endtime_sign_check - starttime_sign_check), 'seconds for resonant sign check\n')
        sign = signs[responses.index(max(responses))]
        if sign < 0:
            output_sign = 'n'

        if input_sign != output_sign:
            flipped = True
        else:
            flipped = False
        return sign, flipped


    def _initialize_influence_functions(self, decoupled_influences):
        '''
        calculates the influences of each load direction at each node 
        if decoupled_influences = True: 
            influences are caclulated simple/manually
        else:
            a static analysis with a respective unit load is created 
        '''
        # the influences on the base moments are needed in the resonant computations, thus they are computed at least for all of them
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

    # # PLOTS

    def plot_eswl_directional_components(self, load_directions, response_label):
        
        plotter_utilities.plot_load_components(self.eswl_total, 
                                                self.structure_model.nodal_coordinates,
                                                load_directions,
                                                response_label)

    def plot_eswl_components(self, response_label, load_directions, textstr, influences , components_to_plot = ['all']):

        plotter_utilities.plot_eswl_components(self.eswl_components,
                                                self.structure_model.nodal_coordinates,
                                                load_directions,
                                                response_label,
                                                textstr,
                                                influences,
                                                components_to_plot)

