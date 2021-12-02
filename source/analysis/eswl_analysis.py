import numpy as np

from source.analysis.analysis_type import AnalysisType
from source.model.structure_model import StraightBeam
from source.ESWL.ESWL import ESWL

import source.auxiliary.global_definitions as GD
import source.ESWL.eswl_auxiliaries as auxiliary
from source.auxiliary.other_utilities import get_adjusted_path_string
import source.ESWL.eswl_plotters as eplt
import source.postprocess.plotter_utilities as plotter_utilities


class EswlAnalysis(AnalysisType):

    def __init__(self, structure_model, parameters):
        '''
        eigenvalue_analysis.eigenform is required for RESWL
        dynamic_analysis results for postprocess
        '''
        #Validate and assign defaults
        self.structure_model = structure_model
        self.parameters = parameters
        self.settings = parameters['settings']
        self.plot_parameters = parameters['output']['plot']
        
        super().__init__(structure_model, self.parameters["type"])#?needed

    def solve(self):


        # drop the first entries beloning to the ramp up
        load_signals_raw = np.load(get_adjusted_path_string(self.parameters['input']['file_path']))

        if len(load_signals_raw) != (self.structure_model.n_nodes*GD.DOFS_PER_NODE[self.structure_model.domain_size]):
            raise Exception('beam model and dynamic load signal have different number of nodes')
        else:
            # PARSING FOR ESWL
            time_info = self.parameters['input']['time_info']
            load_signals = auxiliary.parse_load_signal(load_signals_raw, time_info, GD.DOFS_PER_NODE[self.structure_model.domain_size])

        self.eswl = ESWL(self.structure_model, self.settings, load_signals)

        for response in self.settings['responses_to_analyse']:

            print ('\nCalculating ESWL for', response)

            self.eswl.calculate_total_ESWL(response)

            # ===============================================
            # RUN A STATIC ANALYSIS WITH THE ESWL
            # ===============================================
            print('\nStatic analysis with ESWL...')
            self.eswl.evaluate_equivalent_static_loading()

            print()

    def postprocess(self, global_folder_path, pdf_report, display_plot, skin_model_params):
        
        if self.plot_parameters['influence_functions']:
            plotter_utilities.plot_influences(eswl)

        if self.plot_parameters['mode_shapes']:
            self.eigenform_sorted = self.sort_row_vectors_dof_wise(self.eigenform_unsorted)
            plotter_utilities.plot_n_mode_shapes(self.eigenform_sorted, self.structure_model.charact_length)

        if self.plot_parameters['eswl_load_distribution']['plot']:
            for response in self.settings['responses_to_analyse']:
                plotter_utilities.plot_eswl_components(self.eswl.eswl_components, response, 
                            self.plot_parameters['eswl_load_distribution'], display_plot)

        if self.plot_parameters['eswl_component_rate']:
            plotter_utilities.plot_component_rate(self.eswl, display_plot)