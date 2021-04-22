from .serialization import json_serialization_classifiers as jcl
from .serialization import json_serialization_regressors as jrg
from .serialization import json_serialization_clustering as jct
import json

def deserialized_datrics_model(json_model):
    datrics_model = None
    try:
        if (len(json_model) > 0) and type(json_model == dict) and ("meta" in json_model.keys()):
            if json_model["meta"] in ["Logistic_regression", "LightGBM_binary_classificator",
                                      "LightGBM_multiclass_classificator"]:
                datrics_model = jcl.deserealized_classifier(json_model)
            if json_model["meta"] in ["LinearRegressor", "LightGBM_regressor"]:
                datrics_model = jrg.deserealized_regressor(json_model)
            if json_model["meta"] in ["KMeans_segmentation", "IsolationForest"]:
                datrics_model = jct.deserealized_clustering(json_model)
            else:  # TODO add another models
                pass
    except Exception:
        datrics_model = None
    return datrics_model

def from_json(model_name):
    with open(model_name, 'r') as model_json:
        model_dict = json.load(model_json)
        return deserialized_datrics_model(model_dict)

def from_dict(model_dict):
    return deserialized_datrics_model(model_dict)