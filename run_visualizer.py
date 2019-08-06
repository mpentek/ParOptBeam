import json
import os
from source.postprocess.visualize_skin_model_utilities import visualize_skin_model

json_file_name = os.path.join(*["input","parameters","ProjectParameters3DSkinVisualizationUtility.json"])
visualize_skin_model(json_file_name)