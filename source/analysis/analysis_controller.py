from os.path import join, isdir
from os import makedirs
from matplotlib.backends.backend_pdf import PdfPages

from source.model.structure_model import StraightBeam
from source.auxiliary.validate_and_assign_defaults import validate_and_assign_defaults


class AnalysisController(object):
    """
    Dervied class for the dynamic analysis of a given structure model        

    """

    POSSIBLE_ANALYSES = ['eigenvalue_analysis',
                         'dynamic_analysis',
                         'static_analysis']

    # using these as default or fallback settings
    DEFAULT_SETTINGS = {
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

        # TODO: some more robust checks and assigns
        if self.parameters['report_options']['combine_plots_into_pdf']:
            file_name = 'analyses_results_report.pdf'
            absolute_folder_path = join("output", self.model.name)
            # make sure that the absolute path to the desired output folder exists
            if not isdir(absolute_folder_path):
                makedirs(absolute_folder_path)

            self.report_pdf = PdfPages(
                join(absolute_folder_path, file_name))
        else:
            self.report_pdf = None

        self.display_plots = self.parameters['report_options']['display_plots_on_screen']

        # TODO: passing skin model visualisation params
        self.skin_model_params = None
        if self.parameters['report_options']['use_skin_model']:
            self.skin_model_params = {"geometry": self.parameters["skin_model_parameters"]["geometry"],
                                      "length": self.model.parameters["lx"],
                                      "contour_density": self.parameters["skin_model_parameters"]["contour_density"],
                                      "record_animation": self.parameters["skin_model_parameters"]["record_animation"],
                                      "visualize_line_structure": self.parameters["skin_model_parameters"][
                                          "visualize_line_structure"],
                                      "beam_direction": self.parameters["skin_model_parameters"]["beam_direction"],
                                      "scaling_vector": self.parameters["skin_model_parameters"]["scaling_vector"],
                                      "num_of_dofs_per_node": self.model.DOFS_PER_NODE[self.model.domain_size],
                                      "dofs_input": {}}

        self.analyses = []

        for analysis_param in parameters['runs']:
            if analysis_param['type'] == 'eigenvalue_analysis':
                from source.analysis.eigenvalue_analysis import EigenvalueAnalysis
                self.analyses.append(EigenvalueAnalysis(
                    self.model, analysis_param))
                pass

            elif analysis_param['type'] == 'dynamic_analysis':
                from source.analysis.dynamic_analysis import DynamicAnalysis
                self.analyses.append(DynamicAnalysis(
                    self.model, analysis_param))

            elif analysis_param['type'] == 'static_analysis':
                from source.analysis.static_analysis import StaticAnalysis
                self.analyses.append(StaticAnalysis(
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
        self.model.plot_model_properties(self.report_pdf, self.display_plots, self.skin_model_params)

        for analysis in self.analyses:
            analysis.postprocess(self.report_pdf, self.display_plots, self.skin_model_params)

        try:
            self.report_pdf.close()
        except:
            pass
