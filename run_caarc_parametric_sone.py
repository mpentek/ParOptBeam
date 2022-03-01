from os.path import join
import json
from numpy import loadtxt, tan, savetxt, c_, array, isinf, isnan, rad2deg, cos, pi

from source.model.structure_model import StraightBeam
from source.analysis.analysis_controller import AnalysisController

def extrapolate_data(disp, rot, x_cur=120.0, x_tar=180.0):
    # '''
    # modified exponential
    # to include the effect of rotation
    # '''
    # if abs(disp) < 1e-3:
    #     b = 0
    # else:
    #     deg_rot = rad2deg(rot)
    #     tan_rot = tan(rot)
    #     b = x_cur * tan(tan_rot) / disp
    # a = disp / (x_cur/x_tar)**b
    # # the line below is a * 1.0. 
    # # the influence of rotation and displacement 
    # # is already contained
    # val = a*(x_tar/x_tar)**b 
    # if isinf(val):
    #     print(val)
    # elif isnan(val):
    #     print(val)
    # elif abs(val) > 1e5:
    #     print(val)
    #     pass

    # NOTE: for now changing to harmonic 
    # as this seems to have problems with disp and rot close to zero

    a = disp / (1 - cos(pi/2 * x_cur / x_tar)) 
    val = a * (1-cos(pi/2 * x_tar/x_tar))

    return val

# ==============================================
# Model choice

# NOTE: all currently available files
parametric_runs = {
    'input/force/caarc/0_turb/force_dynamic_0_turb':
        {'output_folder_prefix': 'turb',
         'project_params':
            ['ProjectParameters3DCaarcBeamCont0.json',
             'ProjectParameters3DCaarcBeamInt0.json',
             'ProjectParameters3DCaarcBeamIntOut0.json']},
    'input/force/caarc/45_turb/force_dynamic_45_turb':
        {'output_folder_prefix': 'turb',
         'project_params':
            ['ProjectParameters3DCaarcBeamCont45.json',
             'ProjectParameters3DCaarcBeamInt45.json',
             'ProjectParameters3DCaarcBeamIntOut45.json']},
    'input/force/caarc/90_turb/force_dynamic_90_turb':
        {'output_folder_prefix': 'turb',
         'project_params':
            ['ProjectParameters3DCaarcBeamCont90.json',
             'ProjectParameters3DCaarcBeamInt90.json',
             'ProjectParameters3DCaarcBeamIntOut90.json']},  
             # 30001 expected, 21471 available  
    # 'input/force/caarc/0_no_turb/force_dynamic_0_no_turb':
    #     {'output_folder_prefix': 'no_turb',
    #      'project_params':
    #         ['ProjectParameters3DCaarcBeamCont0.json',
    #          'ProjectParameters3DCaarcBeamInt0.json',
    #          'ProjectParameters3DCaarcBeamIntOut0.json']},    
    'input/force/caarc/45_no_turb/force_dynamic_45_no_turb':
        {'output_folder_prefix': 'no_turb',
         'project_params':
            ['ProjectParameters3DCaarcBeamCont45.json',
             'ProjectParameters3DCaarcBeamInt45.json',
             'ProjectParameters3DCaarcBeamIntOut45.json']},
    'input/force/caarc/90_no_turb/force_dynamic_90_no_turb':
        {'output_folder_prefix': 'no_turb',
         'project_params':
            ['ProjectParameters3DCaarcBeamCont90.json',
             'ProjectParameters3DCaarcBeamInt90.json',
             'ProjectParameters3DCaarcBeamIntOut90.json']}}

# ==============================================
# Parametric run

for damping_ratio in ['0.000', '0.01', '0.025', '0.05']:
    for load_file, parametric_run in parametric_runs.items():

        for available_model in parametric_run['project_params']:

            # ==============================================
            # Parameter read
            with open(join(*['input', 'parameters', 'caarc', available_model]), 'r') as parameter_file:
                parameters = json.loads(parameter_file.read())

            for n_el in [1]: #[60, 30, 15, 3, 2, 1]:
                parameters["model_parameters"]["system_parameters"]["geometry"]["number_of_elements"] = n_el
                parameters["model_parameters"]["system_parameters"]["material"]["damping_ratio"] = float(
                    damping_ratio)

                # manually set length to 120.0
                parameters["model_parameters"]["system_parameters"]["geometry"]["length_x"] = 120.0
                
                # 1st try
                # manually increase target total mass such that effect modal mass in 1st mode gets comparable
                if parameters["model_parameters"]["name"][:-1] == "CaarcBeamCont":
                    parameters["optimization_parameters"]["adapt_for_target_values"]["density_for_total_mass"] = 38880000.0 *1.25

                # 2nd try
                # manually increase target total mass such that effect modal mass for all models
                if parameters["model_parameters"]["name"][:-1] == "CaarcBeamCont": # seems to need larger increase
                    parameters["optimization_parameters"]["adapt_for_target_values"]["density_for_total_mass"] = 38880000.0 *2.5
                else: # seem to need less increase
                    parameters["optimization_parameters"]["adapt_for_target_values"]["density_for_total_mass"] = 38880000.0 *2.0

                # 3rd try
                # manually increase target total mass such that effect modal mass for all models
                if parameters["model_parameters"]["name"][:-1] == "CaarcBeamCont": # seems to need larger increase
                    parameters["optimization_parameters"]["adapt_for_target_values"]["density_for_total_mass"] = 38880000.0 *3.5
                else: # seem to need less increase
                    parameters["optimization_parameters"]["adapt_for_target_values"]["density_for_total_mass"] = 38880000.0 *2.75


                global_output_folder = join(*['Caarc',
                                              'spatial',
                                              parametric_run['output_folder_prefix'],
                                              damping_ratio.replace('.', '_'),
                                              parameters['model_parameters']['name'],
                                              #str(n_el)
                                              'sone'
                                              ])

                parameters["analyses_parameters"]["global_output_folder"] = global_output_folder

                for idx, parameter in enumerate(parameters["analyses_parameters"]["runs"]):
                    if parameter["type"] == "dynamic_analysis":
                        #load_file_path = load_file + str(n_el + 1) + '.npy'
                        load_file_path = load_file + '_sone' + '.npy'
                        load_file_path = join(*load_file_path.split('/'))
                        parameters["analyses_parameters"]["runs"][idx]["input"]["file_path"] = load_file_path

                        # add additional output -> beta:b=-2, gamma:g=-1
                        my_settings = {
                            "dof_list": [1, 2, 0, 4, 5, 3,
                                        -5, -1, # disp y and rot g 
                                        -4, -2],# disp z and rot b
                            "help": "result type can be a list containing: reaction, ext_force, displacement, velocity, acceleration",
                            "result_type": [["reaction"], ["reaction"], ["reaction"], ["reaction"], ["reaction"], ["reaction"],
                                            ["displacement", "velocity", "acceleration"], ["displacement", "velocity", "acceleration"], 
                                            ["displacement", "velocity", "acceleration"], ["displacement", "velocity", "acceleration"]],
                            "plot_result": [[True], [True], [True], [True], [True], [True],
                                            [True, True, True], [True, True, True], 
                                            [True, True, True], [True, True, True]],
                            "write_result": [[False],[False],[False],[True],[True],[True],
                                                [True, False, True], [True, False, True], 
                                                [True, False, True], [True, False, True]]
                        }

                        parameters["analyses_parameters"]["runs"][idx]["output"]["selected_dof"] = my_settings

                # create initial model
                beam_model = StraightBeam(parameters['model_parameters'])

                # additional changes due to optimization
                if 'optimization_parameters' in parameters:
                    # return the model of the optimizable instance to preserve what is required by analyzis
                    from source.model.optimizable_structure_model import OptimizableStraightBeam
                    beam_model = OptimizableStraightBeam(
                        beam_model, parameters['optimization_parameters']['adapt_for_target_values']).model
                else:
                    print('No need found for adapting structure for target values')

                # ==============================================
                # Analysis wrapper

                analyses_controller = AnalysisController(
                    beam_model, parameters['analyses_parameters'])
                analyses_controller.solve()
                analyses_controller.postprocess()

                # ==============================================
                # Extrapolate for top value
                cases = {
                    'xy_plane': {'disp': -5, 'rot': -1},
                    'xz_plane': {'disp': -4, 'rot': -2},
                }

                for key, value in cases.items():

                    print("Computing results for " + key)

                    disp_id = str(value['disp'])
                    rot_id = str(value['rot'])

                    file_pre = 'dynamic_analysis_result_'
                    file_case = ['displacement', 'acceleration']
                    file_suf = '_for_dof_'

                     
                    for fc in file_case:
                        input_file_name = file_pre + fc + file_suf + disp_id + '.dat' 
                        with open(join(*['output',global_output_folder, input_file_name])) as input_file:
                            time_hist, kinem_hist = loadtxt(input_file,delimiter=' ', usecols = (0,1), unpack=True)

                        input_file_name = file_pre + fc + file_suf + rot_id + '.dat' 
                        with open(join(*['output',global_output_folder, input_file_name])) as input_file:
                            kinem_deriv_hist = loadtxt(input_file, usecols = (1,))

                        # NOTE: sign flip for rotation seem to be needed
                        # TODO: check definition
                        kinem_extrapolated_hist = array([extrapolate_data(k,-k_deriv) for k, k_deriv in zip(kinem_hist, kinem_deriv_hist)])

                        output_file_name = file_pre + fc + file_suf + fc + '_' + key + '_top.dat'
                        my_header = 'Dynamic Analysis result ' + fc + '\n'
                        my_header += 'extrapolation in plane ' + key + ' over time'
                        savetxt(join(*['output',global_output_folder, output_file_name]), c_[time_hist, kinem_extrapolated_hist], header=my_header)

                        print("Output file " + str(output_file_name) + " created")
