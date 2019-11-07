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
