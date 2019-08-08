import json
from os.path import join
from source.postprocess.visualize_skin_model_utilities import visualize_skin_model

json_file_name = join(
    *["input", "parameters", "ProjectParameters3DSkinVisualizationUtility.json"])
visualize_skin_model(json_file_name)
