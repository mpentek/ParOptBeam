from visualize_skin_model.Visualiser import *
from visualize_skin_model.LineStructureModel import LineStructure
from visualize_skin_model.StructureModel import Structure

if __name__ == "__main__":
    json_file_name = "trapezoid.json"
    s = Structure(json_file_name)
    ls = LineStructure(json_file_name)
    plotter = Visualiser(s, ls)