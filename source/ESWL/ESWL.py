import numpy as np
from functools import partial
from scipy.optimize import minimize, minimize_scalar

import source.auxiliary.global_definitions as GD
import source.ESWL.eswl_plotters as plotter_utilities
import source.ESWL.eswl_auxiliaries as auxiliary
import source.auxiliary.statistics_utilities as stats_utils

''' 
Main sources and theoretical background:
- summarized in Master Thesis JZimmer
- Chen & Kareem: https://www.researchgate.net/publication/228743607_Equivalent_Static_Wind_Loads_on_Buildings_New_Model 
- Chen & Kareem: https://www.researchgate.net/publication/282252696_Coupled_Dynamic_Wind_Load_Effects_on_Tall_Buildings_with_Three-Dimensional_Modes 
- Kasperski (LRC):  https://www.sciencedirect.com/science/article/abs/pii/0167610592905882 
                    https://www.researchgate.net/publication/264844555_Incorporation_of_the_LRC-method_into_codified_wind_load_distributions 
- Holmes: Wind Loading of Structures 3rd Edit - Chapter 5 (5.4)
''' 
response_labels = ['Qy', 'Qz', 'Mx', 'My', 'Mz']

class ESWL(object):

    def __init__(self, structure_model, settings, load_signals):
        '''
        ESWL object is initialized for the response given (base reactions atm):
        [Qx, Qy, Qz, Mx, My, Mz]
        Attributes:
            - structure_model: an object of the Straight_beam
            - settings
            - load_signals: load signals parsed by auxiliary.parse_load_signal() [as nodal forces and moments]

        '''

        self.structure_model = structure_model
        self.eigenform = structure_model.recuperate_bc_by_extension(structure_model.eigen_modes_raw)
        self.load_signals = load_signals
        self.settings = settings
        
        # beswl settings
        self.beswl_options = self.settings['beswl_options']
        self.reswl_options = self.settings['reswl_options']

        # # do this at initialization since it is needed multiple times and pass it to R and B
        if self.settings['influence_function_computation']:
            print ('\nusing analytic influence functions\n')
        else:
            print ('\nusing influences from static analysis\n')
        self.initialize_influence_functions(self.settings['influence_function_computation'])

        # first key is the response, second the load direciton
        self.eswl_components = {}
        self.static_response = {}

    def calculate_total_ESWL(self, response):
        '''
        Fills a dictionary with the components of the ESWL that are specefied (mean, gle, resonant, total ,lrc)
        '''
        if response in response_labels:
            self.response = response
            self.eswl_components[response] = {}
            self.load_directions_to_compute = GD.LOAD_DIRECTIONS_RESPONSES_UNCOUPLED[response]
        else:
            raise Exception(response + ' is not a valid response label, use: ', response_labels)

        from source.ESWL.BESWL import BESWL
        from source.ESWL.RESWL import RESWL

        # initializing the Background and Resonant Part
        self.BESWL = BESWL(self.structure_model, self.influences, self.load_signals, self.response, self.load_directions_to_compute, self.beswl_options)
        self.RESWL = RESWL(self.structure_model, self.influences, self.load_signals, self.response, self.eigenform, self.reswl_options)

        print ('\nESWL Component Combinations:')
        # calculate the eswl for each load direction

        for direction in self.load_directions_to_compute:
            # # background peak factor with selected method
            g_b = self.get_peak_factor('default')

            self.component_combination(direction, g_b=g_b)

        print ('\nUsing gb:', g_b)

    def component_combination(self, direction, g_b):
        ''' 
        The three components: mean, background, resonant are linearly combined with the use of weighting factors (see Thesis JZ Ch. 5.6).
        The sign of the background and resonant is not uniquely defined by their definitions. Thus the signs are determined here.
        ''' 
        # maximum response
        R_max = self.get_maximum_response(self.response, g_b)

        # =======================================================================
        # # MEAN PART
        mean_load = np.asarray([np.mean(x) for x in self.load_signals[direction]])

        # background is a weighted std, thus it should increase the mean value absolutley or act such that mean + backgr. + res. = maximum
        background_sign = self.get_sign_for_background(mean_load, direction)
        # =======================================================================
        # GLE - KAREEM
        if self.beswl_options['gle']:
            w_b = self.BESWL.weighting_factors_raw_gle[self.response][direction] * g_b / R_max
            p_z_b = self.BESWL.spatial_distribution_gle[self.response][direction] * g_b
            p_z_b_e_gle = abs(w_b * p_z_b) * background_sign

        # #LRC KASPERSIK
        if self.beswl_options['lrc']:
            w_b_lrc = np.sqrt(self.BESWL.weighting_factors_raw_lrc) * g_b / R_max # weighting factor for combinations Kareem eq. 33, Holmes eq.4.41
            p_z_b_lrc = g_b * self.BESWL.spatial_distribution_lrc[self.response][direction]
            p_z_b_e_lrc = abs(w_b_lrc * p_z_b_lrc) * background_sign
     
        # primary use lrc -> if both are true lrc is taken fro combination
        if self.settings['beswl_to_combine'] == 'lrc':
            background = p_z_b_e_lrc
        elif self.settings['beswl_to_combine'] == 'gle':
            background = p_z_b_e_gle

        # =======================================================================
        # # RESWL - RESONANT PART
        p_z_r_all = {}
        p_z_r_e_all = {}
        # NOTE: seperation of modes -> Rule: fi+1*0,9 > fi -> given for CAARC
        w_r_j = self.RESWL.weighting_factors_raw[self.response][direction]
        # 1. variante: distribute resonant base moment along height Kareem eq. 29
        if self.reswl_options['base_moment_distr']:
            p_z_r_all['base_moment_distr'] = self.RESWL.spatial_distribution[self.response][direction]
        # 2. modal inertial load: Kareem eq. 27
        if self.reswl_options['modal_inertial']:
            p_z_r_all['modal_inertial'] = self.RESWL.modal_inertial[self.response][direction]

        types_to_compute_res = [i for i in self.reswl_options if self.reswl_options[i]]
            
        # sum up over first 3 modes and multiply with weighting factor
        for mode_id in range(3):
            g_r = self.get_peak_factor('resonant', self.structure_model.eig_freqs[mode_id])
            w_r = w_r_j[mode_id] * g_r / R_max # weighting factor
            # multiplying p_z_r_ with g_r according to eq. 27,29 Kareem
            for type_r in types_to_compute_res:
                if mode_id == 0:
                    p_z_r_e_all[type_r] = np.zeros(self.structure_model.n_nodes)
                p_z_r_e_all[type_r] +=  w_r * p_z_r_all[type_r][mode_id] * g_r

        # selecting th type for the combination and save it in a extra variable to better handle it
        p_z_r_e = np.copy(p_z_r_e_all[self.settings['reswl_to_combine']])

        # # SIGN OF RESONANT PART
        # 1. select sign du to combination with other components
        resonant_sign, flipped_sign = self.get_sign_for_resonant(mean_load ,background ,p_z_r_e, direction)

        if flipped_sign:
            print ('   flipped sign of resonant', direction, ', reason: component combination')
        
        #if self.reswl_settings['base_moment_distr']:
        p_z_r_e *= resonant_sign

        for type_r  in types_to_compute_res:
            p_z_r_e_all[type_r] *= resonant_sign

        '''
        NOTE: here manually adjusting the signs of b(My), g(Mz) coupled with z(Fz), y(Fy) of resonant component.
        since resonant loads are dependent on the mode shapes and b,z and g,y are coupled the signs must be coupled aswell
        '''
        # coupled g and y
        if direction == 'y':
            self.sign_y = np.sign(p_z_r_e[1])

        if direction == 'g':
            if np.sign(p_z_r_e[1]) != self.sign_y:
                print ('   flipped sign of resonant g, reason: coupled y-g')
                p_z_r_e *= -1
                for type_r  in types_to_compute_res:
                    p_z_r_e_all[type_r] *= -1
        
        # coupled b and z
        if direction == 'z':
            self.sign_z = np.sign(p_z_r_e[1])

        if direction == 'b':
            if np.sign(p_z_r_e[1]) != self.sign_z:
                print ('   flipped sign of resonant b, reason: coupled z-b')
                p_z_r_e *= -1
                for type_r  in types_to_compute_res:
                    p_z_r_e_all[type_r] *= -1

        # =======================================================================
        # # TOTAL
        eswl_total = mean_load + background + p_z_r_e

        # collect all load components in one dict to easy access them
        self.eswl_components['x_coords'] = self.structure_model.nodal_coordinates['x0']
        self.eswl_components[self.response][direction] = {}
        self.eswl_components[self.response][direction]['mean'] = mean_load
        if self.beswl_options['gle']:
            self.eswl_components[self.response][direction]['gle'] = p_z_b_e_gle
        if self.beswl_options['lrc']:
            self.eswl_components[self.response][direction]['lrc'] = p_z_b_e_lrc
        if self.reswl_options['base_moment_distr']:
            self.eswl_components[self.response][direction]['base_moment_distr'] = p_z_r_e_all['base_moment_distr']
        if self.reswl_options['modal_inertial']:
            self.eswl_components[self.response][direction]['modal_inertial'] = p_z_r_e_all['modal_inertial']
        self.eswl_components[self.response][direction]['total'] = eswl_total
               
    def get_peak_factor(self, response_type = None , frequency = None, probability_of_exceedance = 0.98, T = 600):
        '''
        Calculation of peak factor:
        response_type:
            - 'default' or None: fixed to 3.5
            - 'resonant': caculation according a standard formula
                frequency: required - usually eigenfrequency 
            - 'extreme_value': r_max / rms(dynamic response)
                probability_of_exceedance: used for calculating r_max

            T: duration for which the peak factor should be calculated: in DIN this is 600 second = 10 minutes
        '''
        if not response_type:
            return 3.5

        elif response_type == 'default':
            return 3.5

        elif response_type == 'resonant':
            g_r = np.sqrt(2*np.log(T*frequency)) + 0.5772/np.sqrt(2*np.log(T*frequency))
            return g_r

    def get_maximum_response(self, response, g_b):
        '''
        maximum response according to e.g. Holmes eq. 5.41 and Kareem eq. 18
        passing g_b as variable for optimization
        '''
        # background
        #g_b = self._get_peak_factor('gle')#'extreme_value
        R_b = g_b**2 * self.BESWL.rms_background_response

        # resonant
        # for each response one mode is the most relevant, the g_R associated with this mode is then the one mostly contributing to the response
        R_r = 0.0
        for mode_id in range(3):
            sig_R_r = self.RESWL.get_rms_resonant_response(mode_id)
            g_r = self.get_peak_factor('resonant', self.structure_model.eig_freqs[mode_id])
            R_r += g_r**2 * sig_R_r

        return np.sqrt(R_b + R_r)

    def get_sign_for_resonant(self, mean, background, resonant, current_direction):
        '''
        The background component is already added such that it increases the mean part at each node
        The resonant part must either be added or subtracted at a whole, not node wise.
        Thus here changing the sign such that the maximum response is reached
        '''
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

        sign = signs[responses.index(max(responses))]
        if sign*resonant[1] < 0:
            output_sign = 'n'

        if input_sign != output_sign:
            flipped = True
        else:
            flipped = False
        return sign, flipped

    def get_sign_for_background(self, mean_load, current_direction):
        ''' 
        find a sign for the backgorund componetne according to the mean load distribution
        This is relevant if the mean value is small and changes sign along the height. 
        If the background sign is choosen at each point to amplify the respective mean, this creates a favourable situation. 
        However for the nodal moment signals it makes sense to decied it node wise especially concerning the bottom and top node
        '''
        if current_direction in ['b','g']:
            background_sign = np.ones(mean_load.size)
            for node in range(len(mean_load)):
                if mean_load[node] <0:
                    background_sign[node] *= -1
            return background_sign

        if all(item >= 0 for item in mean_load):
            return 1
        if all(item < 0 for item in mean_load):
            return -1
        else:
            influence = self.influences[self.response][current_direction]
            R = sum(np.multiply(influence, mean_load))
            negative_sum = sum(mean_load[mean_load < 0])
            positive_sum = sum(mean_load[mean_load > 0])
            if R > 0:
                return -1
            else:
                return 1

    def initialize_influence_functions(self, method):
        '''
        calculates the influences of each load direction at each node
        method
            'analytic': influences are caclulated simple/manually
        else:
            a static analysis with a respective unit load is created
        '''
        # the influences on the base moments are needed in the resonant computations, thus they are computed at least for all of them
        required_responses = ['Qy', 'Qz', 'Mx', 'My', 'Mz']
        
        h = self.structure_model.nodal_coordinates['x0'][-1]
        self.response_node_id = int(round(self.settings['at_height']/( h / self.structure_model.n_nodes)))

        self.influences = {}
        for response in required_responses:
            self.influences[response] = {}
            for direction in GD.DOF_LABELS['3D']:
                self.influences[response][direction] = np.zeros(self.structure_model.n_nodes)
                for node in range(self.structure_model.n_nodes):
                    if method == 'analytic':
                        influence = self.get_analytic_influences(direction, node, response, self.response_node_id)
                    else:
                        influence = self.get_influence(direction, node, response, response_node_id)
                    self.influences[response][direction][node] = influence

    def get_influence(self, load_direction, node_id, response, response_node_id = 0):
        '''
        influence function representing the response R due to a unit load acting at elevation z along load direction s
        computed using the beam model and a static analysis with load vector of zeros just 1 at node_id
        NOTE: Sofar returning always reaction at ground node
        '''
        src_path = os.path.join(*['input','force','generic_building','unit_loads'])

        needed_force_file = src_path + os.path.sep + 'unit_static_force_' + str(self.structure_model.n_nodes) + \
                            '_nodes_at_' + str(node_id) + \
                            '_in_' + load_direction+'.npy'

        if os.path.isfile(needed_force_file):
            unit_load_file = needed_force_file
        else:
            unit_load_file = auxiliary.generate_unit_nodal_force_file(self.structure_model.n_nodes, node_id, load_direction, 1.0)

        static_analysis = create_static_analysis_custom(self.structure_model, unit_load_file)
        static_analysis.solve()

        influence = static_analysis.reaction[GD.RESPONSE_DIRECTION_MAP[response]]

        if load_direction in ['a','b','g'] and node_id == 0 and load_direction == GD.RESPONSE_DIRECTION_MAP[response]:
            if node_id >= response_node_id:
                return 1.0
            else:
                return 0.0
        # maybe due to numerical stuff or whatever
        # set small values that are mechanically expected to be 0 to actual 0 that in the b_sl calculation 0 and not a radnom value occurs
        if abs(influence[0]) < 1e-05:
            influence[0] = 0.0
            
        return influence[response_node_id]

    def get_analytic_influences(self, load_direction, node_id, response, response_node_id=0):
        '''
        for a lever arm this is simple
        if shear response -> return 1
        if base moment -> return level* 1
        '''
        moment_load = {'y':'Mz', 'z':'My', 'a':'Mx','b':'My', 'g':'Mz'}
        shear_load = {'y':'Qy', 'z':'Qz'}

        nodal_coordinates = self.structure_model.nodal_coordinates['x0']

        if load_direction == 'y':
            if moment_load[load_direction] == response:
                # positive
                if node_id - response_node_id <= 0:
                    return 0.0
                else:
                    return nodal_coordinates[node_id - response_node_id]

            elif shear_load[load_direction] == response:
                if node_id >= response_node_id:
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0

        elif load_direction == 'z':
            if moment_load[load_direction] == response:
                # negative
                if node_id - response_node_id <= 0:
                    return 0.0
                else:
                    return -nodal_coordinates[node_id - response_node_id]
            elif shear_load[load_direction] == response:
                if node_id >= response_node_id:
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0

        elif load_direction == 'x':
            return 0.0

        elif load_direction in ['a','b','g']:
            unit = '[Nm]'
            if moment_load[load_direction] == response:
                if node_id >= response_node_id:
                    return 1.0
                else:
                    return 0.0
            else: # moments don't cause shear forces
                return 0.0

    def evaluate_equivalent_static_loading(self):
        ''' 
        The static response of the structure_model to the ESWL is calculated and saved as self.static_response
        ''' 
        eswl_vector = auxiliary.generate_static_load_vector_file(self.eswl_components[self.response])

        static_analysis = auxiliary.create_static_analysis_custom(self.structure_model, eswl_vector)
        static_analysis.solve()

        self.static_response[self.response] = static_analysis.reaction[GD.RESPONSE_DIRECTION_MAP[self.response]]

