from source.analysis_type import*
#from analysis_type import *

class AnalysisWrapper(object):
    """
    Dervied class for the dynamic analysis of a given structure model        

    """

    POSSIBLE_ANALYSES = ['eigenvalue_analysis', 'dynamic_analysis', 'static_analysis']

    def __init__(self, parameters, model):
        
        self.analyses = []

        for analysis_param in parameters:
            if analysis_param['type'] == 'eigenvalue_analysis':
                self.analyses.append(EigenvalueAnalysis(analysis_param, model))
                pass
            elif analysis_param['type'] == 'dynamic_analysis':
                self.analyses.append(DynamicAnalysis(analysis_param, model))
            elif analysis_param['type'] == 'static_analysis':
                self.analyses.append(StaticAnalysis(analysis_param, model))
            else:
                err_msg = "The analysis type \"" + \
                    analysis_param['type']
                err_msg += "\" is not available \n"
                err_msg += "Choose one of: \""
                err_msg += '\", \"'.join(AnalysisWrapper.POSSIBLE_ANALYSES) + '\"'
                raise Exception(err_msg)