import os

from source.wip_visualize_skin_model.Visualiser import *
from source.wip_visualize_skin_model.LineStructureModel import LineStructure
from source.wip_visualize_skin_model.StructureModel import Structure


def visualize_skin_model(json_file_name):
    s = Structure(json_file_name)
    ls = LineStructure(json_file_name)
    plotter = Visualiser(ls, s)


if __name__ == "__main__":
    json_file_name = "../../input/parameters/ProjectParameters3DSkinVisualizationUtility.json"
    visualize_skin_model(json_file_name)
