from os.path import join, isdir
from os import makedirs
from matplotlib.backends.backend_pdf import PdfPages

from source.model.structure_model import StraightBeam
from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults
from source.auxiliary.other_utilities import get_adjusted_path_string
from source.auxiliary import global_definitions as GD


class AnalysisController(object):
    """
    Dervied class for the dynamic analysis of a given structure model        

    """

    POSSIBLE_ANALYSES = ['eigenvalue_analysis',
                         'dynamic_analysis',
                         'static_analysis']

    # using these as default or fallback settings
    DEFAULT_SETTINGS = {
        "global_output_folder": "some/path",
        "model_properties": {},
        "report_options": {},
        "runs": [],
        "skin_model_parameters": {}}

    def __init__(self, model, parameters):

        if not (isinstance(model, StraightBeam)):
            err_msg = "The proivded model is of type \"" + \
                      str(type(model)) + "\"\n"
            err_msg += "Has to be of type \"<class \'StraigthBeam\'>\""
            raise Exception(err_msg)
        self.model = model

        # validating and assign model parameters
        validate_and_assign_defaults(
            AnalysisController.DEFAULT_SETTINGS, parameters)
        self.parameters = parameters

        if get_adjusted_path_string(self.parameters['global_output_folder']) == get_adjusted_path_string("some/path"):
            self.global_output_folder = join("output", self.model.name)
        else:
            self.global_output_folder = join(
                "output", get_adjusted_path_string(self.parameters['global_output_folder']))

        # make sure that the absolute path to the desired output folder exists
        if not isdir(self.global_output_folder):
            makedirs(self.global_output_folder)

        print(self.global_output_folder +
              ' set as absolute folder path in AnalysisController')

        if self.parameters['report_options']['combine_plots_into_pdf']:
            file_name = 'analyses_results_report.pdf'

            self.report_pdf = PdfPages(
                join(self.global_output_folder, file_name))
        else:
            self.report_pdf = None

        self.display_plots = self.parameters['report_options']['display_plots_on_screen']

        self.skin_model_params = None
        if self.parameters['report_options']['use_skin_model']:
            self.skin_model_params = {"geometry": self.parameters["skin_model_parameters"]["geometry"],
                                      "length": self.model.parameters["lx"],
                                      "record_animation": self.parameters["skin_model_parameters"]["record_animation"],
                                      "visualize_line_structure": self.parameters["skin_model_parameters"][
                                          "visualize_line_structure"],
                                      "beam_direction": self.parameters["skin_model_parameters"]["beam_direction"],
                                      "scaling_vector": self.parameters["skin_model_parameters"]["scaling_vector"],
                                      "num_of_dofs_per_node": GD.DOFS_PER_NODE[self.model.domain_size],
                                      "eigenmode_scaling_factor": self.parameters["skin_model_parameters"][
                                          "eigenmode_scaling_factor"],
                                      "dynamic_scaling_factor": self.parameters["skin_model_parameters"][
                                          "dynamic_scaling_factor"],
                                      "dofs_input": {}}

        self.analyses = []

        for analysis_param in parameters['runs']:
            if analysis_param['type'] == 'eigenvalue_analysis':
                from source.analysis.eigenvalue_analysis import EigenvalueAnalysis
                self.analyses.append(EigenvalueAnalysis(
                    self.model, analysis_param))
                pass

            elif analysis_param['type'] == 'dynamic_analysis':
                # if analysis_param['settings']['run_in_modal_coordinates']:    
                #     from source.analysis.dynamic_analysis import DynamicAnalysis
                #     self.analyses.append(DynamicAnalysis(
                #         self.model, analysis_param))
                # else: 
                print('\nNOTE! Suppressing the dynamic analysis in the analysis_controller\n')
                continue
                from source.analysis.dynamic_analysis import DynamicAnalysis
                self.analyses.append(DynamicAnalysis(
                    self.model, analysis_param))

            elif analysis_param['type'] == 'static_analysis':
                from source.analysis.static_analysis import StaticAnalysis
                self.analyses.append(StaticAnalysis(
                    self.model, analysis_param))
            
            elif analysis_param['type'] == 'ESWL_analysis':
                from source.analysis.eswl_analysis import EswlAnalysis
                self.analyses.append(EswlAnalysis(
                    self.model, analysis_param))

            else:
                err_msg = "The analysis type \"" + \
                          analysis_param['type']
                err_msg += "\" is not available \n"
                err_msg += "Choose one of: \""
                err_msg += '\", \"'.join(
                    AnalysisController.POSSIBLE_ANALYSES) + '\"'
                raise Exception(err_msg)

    def solve(self):
        for analysis in self.analyses:
            analysis.solve()

    def postprocess(self):
        if self.parameters['model_properties']['write']:
            self.model.write_properties(self.global_output_folder)
        if self.parameters['model_properties']['plot']:
            self.model.plot_properties(self.report_pdf, self.display_plots)

        for analysis in self.analyses:
            analysis.postprocess(self.global_output_folder, self.report_pdf,
                                 self.display_plots, self.skin_model_params)

        try:
            self.report_pdf.close()
        except:
            pass
