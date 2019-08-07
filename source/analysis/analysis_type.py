# ===============================================================================
'''
Project:Lecture - Structural Wind Engineering WS17-18 
        Chair of Structural Analysis @ TUM - A. Michalski, R. Wuchner, M. Pentek
        
        Analysis type base class and derived classes specific types

Author: mate.pentek@tum.de, anoop.kodakkal@tum.de, catharina.czech@tum.de, peter.kupas@tum.de

      
Note:   UPDATE: The script has been written using publicly available information and 
        data, use accordingly. It has been written and tested with Python 2.7.9.
        Tested and works also with Python 3.4.3 (already see differences in print).
        Module dependencies (-> line 61-74): 
            python
            numpy
            sympy
            matplotlib.pyplot

Created on:  22.11.2017
Last update: 09.07.2019
'''
# ===============================================================================


class AnalysisType(object):
    """
    Base class for the different analysis types
    """

    def __init__(self, structure_model, name="DefaultAnalysisType"):
        self.name = name

        # the structure model - geometry and physics - has the Dirichlet BC
        # for the bottom node included
        self.structure_model = structure_model

        self.displacement = None
        self.rotation = None

        self.force = None
        self.reaction = None
        self.moment = None

    def solve(self):
        """
        Solve for something
        """
        print("Solving for something in AnalysisType base class \n")
        pass

    def postprocess(self):
        """
        Postprocess something
        """
        print("Postprocessing in AnalysisType base class \n")
        pass

