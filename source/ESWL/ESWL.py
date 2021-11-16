import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import timeit
from functools import partial
from scipy.optimize import minimize, minimize_scalar
from os.path import join as os_join
from os.path import sep

import source.auxiliary.global_definitions as GD
import source.ESWL.eswl_plotters as plotter_utilities
import source.ESWL.eswl_auxiliaries as auxiliary
import source.auxiliary.statistics_utilities as stats_utils
from source.ESWL.eswl_plotters import plot_objective_function_2D, plot_dynamic_results

''' 
Main sources and theoretical background:
- summarized in Master Thesis JZimmer
- Chen & Kareem: https://www.researchgate.net/publication/228743607_Equivalent_Static_Wind_Loads_on_Buildings_New_Model 
- Chen & Kareem: https://www.researchgate.net/publication/282252696_Coupled_Dynamic_Wind_Load_Effects_on_Tall_Buildings_with_Three-Dimensional_Modes 
- Kasperski (LRC):  https://www.sciencedirect.com/science/article/abs/pii/0167610592905882 
                    https://www.researchgate.net/publication/264844555_Incorporation_of_the_LRC-method_into_codified_wind_load_distributions 
- Holmes: Wind Loading of Structures 3rd Edit - Chapter 5 (5.4)
''' 

response_labels = ['Qx', 'Qy', 'Qz', 'Mx', 'My', 'Mz']

class ESWL(object):

    def __init__(self, structure_model, eigenvalue_analysis, response, response_height, load_signals, load_directions, load_directions_to_compute, 
                 decoupled_influences, plot_influences, use_lrc, use_gle, reswl_settings, 
                 optimize_gb = False, target = 'estimate', evaluate_gb = False,
                 plot_mode_shapes = False, dynamic_analysis_solved = None, plot_objective_function = False):
        '''
        ESWL object is initialized for the response given (base reactions atm):
        [Qx, Qy, Qz, Mx, My, Mz]
        Attributes:
            - structure_model: an object of the Straight_beam
            - eigenvalue_analysis: object of eingevalue analysis
            - response: current response for which to calculate the ESWL
            - load_signals: load signals parsed by auxiliary.parse_load_signal() [as nodal forces and moments]
            - load_directions: load directions that are considered in the calculation of correlations between loads 
            - load_directons_to_compute: 
                from [Fx, Fy, Fz, Mx, My, Mz]: actual directions for which the should be computed
                'all': include all directions
                'automatic': relevant load directions for uncoupled self.response 
            - used_lumped: 
                True: if only lumped mass formulations should be used
                False: some parts are formulated in a consistent way
            - decouple_influences:
                True: a simple inlfuence function is computed, no coupling effects are involved here
                False: influence function is calculated with static analyis and unit loads
            - use_lrc: if LRC method shall be used for background part
            - use_gle: if GLE approach should be used for background part
                if both are True LRC will be used for calculation of total ESWL 
            - reswl_settings: different approaches to calculate this part are implemented,
                set booleans for which to calculate and specify which is used for total ESWL
            - inclued_reswl: wheter to include resonant part into calculation of total ESWL
            - optimize_gb: wheter to use an optimization procedure for the background peak factor to reach a specified maximum response ('target')
            - target:
                'default': g_b is not optimized
                'estimate': an extreme Value analysis (estimate) of the dynamic result is set as target max
                'quantile': an extreme Value analysis (quantile) of the dynamic result is set as target max
                            --> difference of these 2 see Extreme Value analysis NIST Translational Process approach
                'max_factor': the global maximum of the dynamic result times a factor 1.3 is set as target
            - plot_mode_shapes: obvious
            - dynamic_analysis_solved: the solver object of a dynamic_analysis after solving, required if optimization of g_b
            - plot_objective_function: wheter to evaluate and plot the objective function of the peak factor adjustment

        '''

        self.structure_model = structure_model
        self.eigenvalue_analysis = eigenvalue_analysis
        self.dynamic_analysis_solved = dynamic_analysis_solved
        self.load_signals = load_signals
        if load_directions == 'all':
            self.load_directions = GD.LOAD_DIRECTION_MAP[load_directions]
        else:
            self.load_directions = [GD.LOAD_DIRECTION_MAP[l_i] for l_i in load_directions]
        if load_directions_to_compute == 'all':
            self.load_directions_to_compute = GD.LOAD_DIRECTION_MAP[load_directions_to_compute]
        elif load_directions_to_compute == 'automatic':
            self.load_directions_to_compute = GD.LOAD_DIRECTIONS_RESPONSES_UNCOUPLED[response]
        else:
            self.load_directions_to_compute = [GD.LOAD_DIRECTION_MAP[l_i] for l_i in load_directions_to_compute]

        self.response_height = response_height
        print ('\nComputing ESWL for load directions', self.load_directions_to_compute, 'for response', response, ' at H = ', self.response_height)
        
        self.use_lrc = use_lrc
        self.use_gle = use_gle
        self.optimize_gb = optimize_gb
        self.target = target
        self.evaluate_gb = evaluate_gb
        self.plot_objective_function = plot_objective_function
        self.reswl_settings = reswl_settings
        self.plot_mode_shapes = plot_mode_shapes

        if response in response_labels:
            self.response = response
        else:
            raise Exception(response + ' is not a valid response label, use: ', response_labels)

        # do this at initialization since it is needed multiple times and pass it to R and B
        self.decoupled_influences = decoupled_influences
        self.plot_influences = plot_influences
        if self.decoupled_influences:
            print ('\nusing decoupled influences\n')
        else:
            print ('\nusing influences from static analysis\n')
        self.initialize_influence_functions(self.decoupled_influences)

        self.eswl_components = {} # first key is the response, second the load direciton
        self.eswl_total = {}

    def calculate_total_ESWL(self):
        '''
        Fills a dictionary with the components of the ESWL that are specefied (mean, gle, resonant, total ,lrc)
        '''
        from source.ESWL.BESWL import BESWL
        from source.ESWL.RESWL import RESWL

        # initializing the Background and Resonant Part
        self.BESWL = BESWL(self.structure_model, self.influences, self.load_signals, self.load_directions, self.response, 
                           self.use_lrc, self.use_gle)
        self.RESWL = RESWL(self.structure_model, self.influences, self.eigenvalue_analysis,
                            self.load_signals, self.load_directions, self.response,
                            self.plot_mode_shapes, self.reswl_settings)

        self.eswl_components[self.response] = {}
        self.eswl_total[self.response] = {}
        
        # setting up the target for the optimization of the factor g_b
        response_id = GD.DOF_LABELS['3D'].index(GD.RESPONSE_DIRECTION_MAP[self.response])
        dynamic_response = self.dynamic_analysis_solved.dynamic_reaction[response_id]

        if self.target == 'default':
            self.optimize_gb = False

        if self.optimize_gb:
            glob_max = max(abs(dynamic_response))
            if self.target == 'max_factor':
                target_response = glob_max*1.3

            elif self.target == 'estimate' or self.target == 'quantile':
                target_response = auxiliary.extreme_value_analysis(self.dynamic_analysis_solved, self.response, self.target)

        self.g_b_optimized = []

        print ('\nESWL Component Combinations:')
        # calculate the eswl for each load direction
        for direction in self.load_directions_to_compute:

            if self.evaluate_gb:
                print ('\nevaluating gb for', self.response, 'in', direction)
                print('WARNING: this cannot be used with optimizing gb in one run sofar')
                resp_gb = []
                for gb_i in np.arange(1, 5.1, 0.1):
                    resp_gb.append(self.component_combination(direction, target_response, gb_i))
                dest = os_join(*['source','ESWL','output','gb_eval'])
                fname = self.response + '_' + direction + '.npy'
                np.save(dest + sep + fname, np.array(resp_gb))
                print('\nsaved:',dest + sep + fname)
                
            # # INCLUDE AN ADAPTION OF THE BACKGROUND PEAK FACTOR FOR A TARGET MAXIMUM RESPONSE
            if self.optimize_gb and not self.target == 'default':
                print ('\nadjusting background peak factor g_b for target dynamic maximum ' + self.target + '...')
                self.objective_function = partial(self.component_combination,
                                                    direction,
                                                    target_response)

                bounds = (1,10)

                min_res = minimize_scalar(self.objective_function, bounds = bounds, method='bounded', options={'disp':True})
                print ('final objective function:', min_res.fun)

                self.g_b_optimized.append(min_res.x)

                if self.plot_objective_function:
                    plot_objective_function_2D(self.objective_function, opt_res=min_res.x,
                                                evaluation_space = [-10,10, 0.5],
                                                design_var_label='g_b for load ' + direction)

                print ('\nadjusted g_b for target static maximum', self.target, self.response, 'for load direction', direction)
                print ('g_b:', min_res.x)
                
            # # BACKGROUND PEAK FACTOR WITH SELECTED METHOD
            else:
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
        background_sign = auxiliary.get_sign_for_background(mean_load, direction)
        # =======================================================================
        # GLE - KAREEM
        if self.use_gle:
            w_b = self.BESWL.weighting_factors_raw[self.response][direction] * g_b / R_max
            p_z_b = self.BESWL.spatial_distribution[self.response][direction] * g_b
            p_z_b_e_gle = abs(w_b * p_z_b) * background_sign

        # #LRC KASPERSIK
        if self.use_lrc:
            w_b_lrc = np.sqrt(self.BESWL.std_background_response_lrc) * g_b / R_max # weighting factor for combinations Kareem eq. 33, Holmes eq.4.41
            p_z_b_lrc = g_b * self.BESWL.get_beswl_LRC(direction)
            p_z_b_e_lrc = abs(w_b_lrc * p_z_b_lrc) * background_sign
     
        # primary use lrc -> if both are true lrc is taken fro combination
        if self.use_lrc:
            background = p_z_b_e_lrc
        elif self.use_gle:
            background = p_z_b_e_gle

        # =======================================================================
        # # RESWL - RESONANT PART
        # NOTE: seperation of modes -> Rule: fi+1*0,9 > fi -> given for CAARC
        w_r_j = self.RESWL.weighting_factors_raw[self.response][direction]
        # 1. variante: distribute resonant base moment along height Kareem eq. 29
        if self.reswl_settings['types_to_compute']['base_moment_distr']:
            p_z_r = self.RESWL.spatial_distribution[self.response][direction]
            p_z_r_e = np.zeros(self.structure_model.n_nodes)
        # 2. modal inertial load: Kareem eq. 27
        if self.reswl_settings['types_to_compute']['modal']:
            p_z_rm = self.RESWL.modal_inertial[self.response][direction]
            p_z_r_em = np.zeros(self.structure_model.n_nodes)
        if self.reswl_settings['types_to_compute']['modal_lumped']:
            p_z_rm_lumped = self.RESWL.modal_inertial_lumped[self.response][direction]
            p_z_r_em_lumped = np.zeros(self.structure_model.n_nodes)
            
        # sum up over first 3 modes and multiply with weighting factor
        for mode_id in range(3):
            g_r = self.get_peak_factor('resonant', self.structure_model.eig_freqs[mode_id])
            w_r = w_r_j[mode_id] * g_r / R_max # weighting factor

            # multiplying p_z_r_ with g_r according to eq. 27,29 Kareem
            if self.reswl_settings['types_to_compute']['base_moment_distr']:
                p_z_r_e +=  w_r * p_z_r[mode_id] * g_r
            if self.reswl_settings['types_to_compute']['modal']:
                p_z_r_em += w_r * p_z_rm[mode_id] * g_r
            if self.reswl_settings['types_to_compute']['modal_lumped']:
                p_z_r_em_lumped += w_r * p_z_rm_lumped[mode_id] * g_r

        # choosing whcihc one is the one to us in the combination 
        if self.reswl_settings['type_to_combine'] == 'base_moment_distr':
            pass
        elif self.reswl_settings['type_to_combine'] == 'modal':
            p_z_r_e = p_z_r_em
        elif self.reswl_settings['type_to_combine'] == 'modal_lumped':
            p_z_r_e = p_z_r_em_lumped

        # # SIGN OF RESONANT PART
        # 1. select sign du to combination with other components
        resonant_sign, flipped_sign = self.get_sign_for_resonant(mean_load ,background ,p_z_r_e, direction)

        if flipped_sign:
            if not self.optimize_gb:
                print ('   flipped sign of resonant', direction, ', reason: component combination')
        
        #if self.reswl_settings['base_moment_distr']:
        p_z_r_e = resonant_sign*p_z_r_e
        if self.reswl_settings['types_to_compute']['modal']:
            p_z_r_em = resonant_sign*p_z_r_em
        if self.reswl_settings['types_to_compute']['modal_lumped']:
            p_z_r_em_lumped = resonant_sign*p_z_r_em_lumped

        '''
        NOTE: here manually adjusting the signs of b(My), g(Mz) coupled with z(Fz), y(Fy) of resonant component.
        since resonant loads are dependent on the mode shapes and b,z and g,y are coupled the signs must be coupled aswell
        '''

        if direction == 'y':
            if p_z_r_e[1] < 0:
                self.sign_y = 'n'
            else:
                self.sign_y = 'p'

        if direction == 'g':
            if p_z_r_e[1] < 0:
                if self.sign_y == 'p':
                    if not self.optimize_gb:
                        print ('   flipped sign of resonant g, reason: coupled y-g')
                    flipped_g_coupling = True
                    p_z_r_e *= -1
                    if self.reswl_settings['types_to_compute']['modal']:
                        p_z_r_em *= -1
                    if self.reswl_settings['types_to_compute']['modal_lumped']:
                        p_z_r_em_lumped *= -1
                else:
                    flipped_g_coupling = False
            else:
                if self.sign_y == 'n':
                    if not self.optimize_gb:
                        print ('   flipped sign of resonant g, reason: coupled y-g')
                    flipped_g_coupling = True
                    p_z_r_e *= -1
                    if self.reswl_settings['types_to_compute']['modal']:
                        p_z_r_em *= -1
                    if self.reswl_settings['types_to_compute']['modal_lumped']:
                        p_z_r_em_lumped *= -1
                else:
                    flipped_g_coupling = False

        if direction == 'z':
            if p_z_r_e[1] < 0:
                self.sign_z = 'n'
            else:
                self.sign_z = 'p'

        if direction == 'b':
            if p_z_r_e[1] < 0:
                if self.sign_z == 'n':
                    if not self.optimize_gb:
                        print ('   flipped sign of resonant b, reason: coupled z-b')
                    flipped_b_coupling =True
                    p_z_r_e *= -1
                    if self.reswl_settings['types_to_compute']['modal']:
                        p_z_r_em *= -1
                    if self.reswl_settings['types_to_compute']['modal_lumped']:
                        p_z_r_em_lumped *= -1
                else:
                    flipped_b_coupling = False
            else:
                if self.sign_z == 'p':
                    if not self.optimize_gb:
                        print ('   flipped sign of resonant b, reason: coupled z-b')
                    flipped_b_coupling = True
                    p_z_r_e *= -1
                    if self.reswl_settings['types_to_compute']['modal']:
                        p_z_r_em *= -1
                    if self.reswl_settings['types_to_compute']['modal_lumped']:
                        p_z_r_em_lumped *= -1
                else:
                    flipped_b_coupling = False

        # =======================================================================
        # # TOTAL
        eswl_total = mean_load + background

        # this is needed in this format for postprocessing
        self.eswl_total[self.response][direction] = eswl_total

        # collect all load components in one dict to easy access them
        self.eswl_components['x_coords'] = self.structure_model.nodal_coordinates['x0']
        self.eswl_components[self.response][direction] = {}
        self.eswl_components[self.response][direction]['mean'] = mean_load
        if self.use_gle:
            self.eswl_components[self.response][direction]['gle'] = p_z_b_e_gle
        if self.use_lrc:
            self.eswl_components[self.response][direction]['lrc'] = p_z_b_e_lrc
        self.eswl_components[self.response][direction]['resonant'] = p_z_r_e
        if self.reswl_settings['types_to_compute']['modal']:
            self.eswl_components[self.response][direction]['resonant_m'] = p_z_r_em
        if self.reswl_settings['types_to_compute']['modal_lumped']:
            self.eswl_components[self.response][direction]['resonant_m_lumped'] = p_z_r_em_lumped
        self.eswl_components[self.response][direction]['total'] = eswl_total

        if self.evaluate_gb:
            self.evaluate_equivalent_static_loading()

            return self.static_response[0]
        
        if self.optimize_gb:
            # evaluate result 
            self.evaluate_equivalent_static_loading()

            current_response = self.static_response[0]

            difference = (abs(target_dyn) - abs(current_response))**2 / target_dyn**2

            return difference
               
    def evaluate_equivalent_static_loading(self):
        ''' 
        The static response of the structure_model to the ESWL is calculated and saved as self.static_response
        ''' 
        eswl_vector = auxiliary.generate_static_load_vector_file(self.eswl_total[self.response])

        static_analysis = auxiliary.create_static_analysis_custom(self.structure_model, eswl_vector)
        static_analysis.solve()

        self.static_response = static_analysis.reaction[GD.RESPONSE_DIRECTION_MAP[self.response]]

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

        elif response_type == 'extreme_value':
            if self.response not in GD.RESPONSE_DIRECTION_MAP.keys():
                raise Exception('peak factor calculations not available for other then base reactions')
            response_id = GD.DOF_LABELS['3D'].index(GD.RESPONSE_DIRECTION_MAP[self.response])
            dynamic_response = self.dynamic_analysis_solved.dynamic_reaction[response_id]
            dt = self.dynamic_analysis_solved.dt
            T_series = self.dynamic_analysis_solved.dt * len(dynamic_response)
            dur_ratio = T / T_series
            # # MAXMINEST NIST
            P1 = probability_of_exceedance
            max_qnt, min_qnt, max_est, min_est, max_std, min_std = stats_utils.maxmin_qnt_est(dynamic_response, cdf_p_max = P1 , cdf_p_min = 0.0001, cdf_qnt = P1, dur_ratio = dur_ratio)
            
            r_max = max([abs(max_est), abs(min_est)])[0]
            rms = stats_utils.rms(dynamic_response)
            
            g = r_max / rms
            print ('\nFor the response', self.response, 'a peak value calculated including mean, resonant and background parts is:')
            print ('  ', round(g,2))
            return g
            # z.B. extreme value analysis or as Kaspersik 2009

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
        if sign < 0:
            output_sign = 'n'

        if input_sign != output_sign:
            flipped = True
        else:
            flipped = False
        return sign, flipped

    def optimize_g_b(self):
        ''' 
        Finding a value of g_b so that the ESWL give a certain target
        TODO this is in calculate_total_eswl sofar
        ''' 


    def initialize_influence_functions(self, decoupled_influences):
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
        
        h = self.structure_model.nodal_coordinates['x0'][-1]
        response_node_id = int(round(self.response_height/(h/self.structure_model.n_nodes)))

        from source.ESWL.eswl_auxiliaries import get_decoupled_influences
        from source.ESWL.eswl_auxiliaries import get_influence

        self.influences = {}
        for response in required_responses:
            self.influences[response] = {}
            for direction in GD.DOF_LABELS['3D']:
                self.influences[response][direction] = np.zeros(self.structure_model.n_nodes)
                for node in range(self.structure_model.n_nodes):
                    if self.decoupled_influences:

                        influence = get_decoupled_influences(self.structure_model, direction, node, response, response_node_id)
                    else:

                        influence = get_influence(self.structure_model, direction, node, response, response_node_id)
                    self.influences[response][direction][node] = influence
        
        if self.plot_influences:
            plotter_utilities.plot_influences(self)

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

