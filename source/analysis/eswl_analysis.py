import numpy as np

from source.analysis.analysis_type import AnalysisType
from source.model.structure_model import StraightBeam
import source.auxiliary.global_definitions as GD
import source.ESWL.eswl_auxiliaries as auxiliary
import source.ESWL.eswl_plotters as eplt
from source.ESWL.ESWL import ESWL

class EswlAnalysis(AnalysisType):

    def __init__(self, structure_model, parameters):
        '''
        eigenvalue_analysis.eigenform is required for RESWL
        dynamic_analysis results for postprocess
        '''
        #Validate and assign defaults
        self.parameters = parameters
        self.settings = parameters['settings']
        
        super().__init__(structure_model, self.parameters["type"])

    def solve(self):

        eigenform = self.structure_model.recuperate_bc_by_extension(
            self.structure_model.eigen_modes_raw)

        for response in self.settings['responses_to_analyse']:
            #eswl = ESWL(self.structure_model, self.parameters, eigenform)
            # eswl = ESWL(beam_model, eigenvalue_analysis, response, response_height, load_signals, load_directions_to_include, load_directions_to_compute,
            #     decoupled_influences, plot_influences, use_lrc, use_gle, reswl_settings, 
            #     optimize_gb, target, evaluate_gb,
            #     plot_mode_shapes, dynamic_analysis.solver, plot_objective_function)

            # if plot_influences:
            #     plotter_utilities.plot_influences(eswl)

            # eswl.calculate_total_ESWL()

            # # ===============================================
            # # RUN A STATIC ANALYSIS WITH THE ESWL
            # # ===============================================
            # print('\nStatic analysis with ESWL...')
            # eswl.evaluate_equivalent_static_loading()

            print()

    def postprocess(self):
        pass