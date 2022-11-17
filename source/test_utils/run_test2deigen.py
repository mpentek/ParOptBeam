from os.path import join as os_join
import json

from source.model.structure_model import StraightBeam
from source.analysis.analysis_controller import AnalysisController

# inputs
parameters = {}

parameters["model_parameters"] = {
        "name": "GenericBuilding",
        "domain_size": "2D",
        #"boundary_conditions": "fixed-free"}
        #"boundary_conditions": "free-fixed"}
        #"boundary_conditions": "fixed-pinned"}
        "boundary_conditions": "pinned-fixed"}

parameters["model_parameters"]["system_parameters"]={
            "element_params": {
                "type": "Timoshenko",
                "is_nonlinear": False
            }}

parameters["model_parameters"]["system_parameters"]["material"] = {
                "is_nonlinear": False,
                "density": 160.0,
                "youngs_modulus": 2.861e8,
                "poisson_ratio": 0.1,
                "damping_ratio": 0.0
            }

parameters["model_parameters"]["system_parameters"]["geometry"] = {
                "length_x": 180,
                "number_of_elements": 3,
                "defined_on_intervals": [{
                    "interval_bounds" : [0.0, "End"],
                    "length_y": [45.0],
                    "length_z": [30.0],
                    "area"    : [1350],
                    "shear_area_y" : [1125.0],
                    "shear_area_z" : [1125.0],
                    "moment_of_inertia_y" : [101250.0],
                    "moment_of_inertia_z" : [227813.0],
                    "torsional_moment_of_inertia" : [329063.0]
                    }]
            }


# parameters["optimization_parameters"] =  {
#         "adapt_for_target_values": {
#             "density_for_total_mass": 38880000.0,
#             "geometric_properties_for": {
#                 "help": "first entry: sway_y, second entry: sway_z, -1: shear, +1: bending",
#                 "partition_shear_bending": [-1, 1],
#                 "consider_decomposed_modes": ["sway_z","sway_y", "torsional"],
#                 "corresponding_mode_ids" : [1, 1, 1],
#                 "corresponding_eigenfrequencies": [0.23,0.20,0.40]}}
#     }
parameters["analyses_parameters"] = {
        "global_output_folder" : "some/path",
        "model_properties": {
            "write": True,
            "plot":True
        }}
parameters["analyses_parameters"]["report_options"] = {
            "combine_plots_into_pdf" : True,
            "display_plots_on_screen" : False,
            "use_skin_model" : True
        }
parameters["analyses_parameters"]["skin_model_parameters"] = {
            "geometry": [ [0, -22.5, -15.0], [0, -22.5, 15], [0, 22.5, 15],
                          [0, 22.5, -15]
            ],
            "contour_density": 1,
            "record_animation": False,
            "visualize_line_structure": False,
            "beam_direction": "x",
            "scaling_vector": [1.0, 1.0 , 1.0, 1.0, 1.0, 1.0, 1.0],
            "eigenmode_scaling_factor" : 1e5,
            "dynamic_scaling_factor" : 1e3
        }

parameters["analyses_parameters"]["runs"] = [{
                "type": "eigenvalue_analysis",
                "settings": {
                    "normalization": "mass_normalized"},
                "input":{},
                "output":{
                    "eigenmode_summary": {
                        "write" : True,
                        "plot" : True},
                    "eigenmode_identification": {
                        "write" : True,
                        "plot" : True},
                    "selected_eigenmode": {
                        "plot_mode": [1,2,3],
                        "write_mode": [1,2,3],
                        "animate_mode": [],
                        "animate_skin_model": []},
                    "selected_eigenmode_range": {
                        "help": "maximum 4 modes per range at a time",
                        "considered_ranges": [[1,2]],
                        "plot_range": [True, True],
                        "write_range": [True, False]}
                    }
            }]

# parameters["analyses_parameters"]["runs"].append(dict({"type" : "dynamic_analysis",
#                 "settings": {
#                     "solver_type": "Linear",
#                     "run_in_modal_coordinates": False,
#                     "time":{
#                         "integration_scheme": "GenAlpha",
#                         "start": 0.0,
#                         "end": 600.0,
#                         "step" : 0.02},
#                     "intial_conditions": {
#                         "displacement": "None",
#                         "velocity": "None",
#                         "acceleration" : "None"
#                     }},
#                 "input": {
#                     "help":"provide load file in the required format",
#                     "file_path": "input/force/generic_building/dynamic_force_4_nodes.npy"
#                 },
#                 "output":{
#                     "selected_instance": {
#                         "plot_step": [1500, 2361],
#                         "write_step": [3276],
#                         "plot_time": [30.5, 315.25],
#                         "write_time": [450.15]
#                     },
#                     "animate_time_history" : True,
#                     "animate_skin_model_time_history": True,
#                     "kinetic_energy": {
#                         "write": True,
#                         "plot": True
#                     },
#                     "skin_model_animation_parameters":{
#                         "start_record": 160,
#                         "end_record": 200,
#                         "record_step": 10
#                     },
#                     "selected_dof": {
#                         "dof_list": [1, 2, 0, 4, 5, 3,
#                                     -5,
#                                     -4,
#                                     -2,
#                                     -1],
#                         "help": "result type can be a list containing: reaction, ext_force, displacement, velocity, acceleration",
#                         "result_type": [["reaction"], ["reaction"], ["reaction"], ["reaction"], ["reaction"], ["reaction"],
#                                         ["displacement", "velocity", "acceleration"],
#                                         ["displacement", "velocity", "acceleration"],
#                                         ["displacement", "velocity", "acceleration"],
#                                         ["displacement", "velocity", "acceleration"]],
#                         "plot_result": [[True], [True], [True], [True], [True], [True],
#                                         [True, True, True],
#                                         [True, True, True],
#                                         [True, False, True],
#                                         [True, False, True]],
#                         "write_result": [[False],[False],[False],[True],[True],[True],
#                                             [True, False, True],
#                                             [True, False, True],
#                                             [True, False, True],
#                                             [True, False, True]]
#                     }
#                 }
#             }))

# parameters["analyses_parameters"]["runs"].append(dict({
#                 "type" : "static_analysis",
#                 "settings": {},
#                 "input":{
#                     "help":"provide load file in the required format - either some symbolic generated or time step from dynamic",
#                     "file_path": "input/force/generic_building/dynamic_force_4_nodes.npy",
#                     "is_time_history_file" : True,
#                     "selected_time_step" : 15000
#                 },
#                 "output":{
#                     "plot": ["deformation", "forces"],
#                     "write": ["deformation"],
#                     "selected_dof": {
#                         "dof_list": [0, 1, 2, 3, 4, 5,
#                                      -6, -5, -4, -3, -2, -1],
#                         "help": "result type can be a list containing: reaction, force, displacement",
#                         "result_type": [["reaction", "force"],
#                                         ["reaction", "force"],
#                                         ["reaction", "force"],
#                                         ["reaction", "force"],
#                                         ["reaction", "force"],
#                                         ["reaction", "force"],
#                                         ["force", "displacement"],
#                                         ["force", "displacement"],
#                                         ["force", "displacement"],
#                                         ["force", "displacement"],
#                                         ["force", "displacement"],
#                                         ["force", "displacement"]],
#                         "write_result": [[True, True],
#                                          [True, True],
#                                          [True, True],
#                                          [True, True],
#                                          [True, True],
#                                          [True, True],
#                                          [True, True],
#                                          [True, True],
#                                          [True, True],
#                                          [True, True],
#                                          [True, True],
#                                          [True, True]]
#                     }
#                 }
#             }))


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
