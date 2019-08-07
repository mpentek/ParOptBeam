from source.postprocess.skin_model.Visualiser import Visualiser
from source.postprocess.skin_model.LineStructureModel import LineStructure
from source.postprocess.skin_model.StructureModel import Structure


def visualize_skin_model(json_file_name):
    s = Structure(json_file_name)
    ls = LineStructure(json_file_name)
    plotter = Visualiser(ls, s)

