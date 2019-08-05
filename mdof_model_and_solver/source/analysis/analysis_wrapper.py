from source.model.structure_model import StraightBeam


class AnalysisWrapper(object):
    """
    Dervied class for the dynamic analysis of a given structure model        

    """

    POSSIBLE_ANALYSES = ['eigenvalue_analysis',
                         'dynamic_analysis', 'static_analysis']

    def __init__(self, model, parameters):

        if not(isinstance(model, StraightBeam)):
            err_msg = "The proivded model is of type \"" + \
                str(type(model)) + "\"\n"
            err_msg += "Has to be of type \"<class \'StraigthBeam\'>\""
            raise Exception(err_msg)

        self.analyses = []

        for analysis_param in parameters:
            if analysis_param['type'] == 'eigenvalue_analysis':
                from source.analysis.eigenvalue_analysis import EigenvalueAnalysis
                self.analyses.append(EigenvalueAnalysis(model, analysis_param))
                pass

            # elif analysis_param['type'] == 'dynamic_analysis':
            #     from source.analysis.dynamic_analysis import DynamicAnalysis
            #     self.analyses.append(DynamicAnalysis(model, analysis_param))

            # elif analysis_param['type'] == 'static_analysis':
            #     from source.analysis.static_analysis import StaticAnalysis
            #     self.analyses.append(StaticAnalysis(model, analysis_param))

            # else:
            #     err_msg = "The analysis type \"" + \
            #         analysis_param['type']
            #     err_msg += "\" is not available \n"
            #     err_msg += "Choose one of: \""
            #     err_msg += '\", \"'.join(AnalysisWrapper.POSSIBLE_ANALYSES) + '\"'
                # raise Exception(err_msg)

    def solve(self):
        for analysis in self.analyses:
            analysis.solve()

    def postprocess(self):
        for analysis in self.analyses:
            analysis.postprocess()
