from visualise_model import *
from NodeModel import Node
from LineStructureModel import LineStructure
from StructureModel import Structure, Floor
from visualise_model import Visualiser

if __name__ == "__main__":
    json_file_name = "trapezoid.json"
    s = Structure(json_file_name)
    ls = LineStructure(json_file_name)
    plotter = Visualiser(s, ls)