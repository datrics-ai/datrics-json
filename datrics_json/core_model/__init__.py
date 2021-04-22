import json

from . import classification as clf
from . import regression as reg
from . import unsupervized as clst

__version__ = "0.1.0"

def deserialize_model(model_dict):
    print (model_dict["meta"])
    if model_dict["meta"] == "lr":
        return clf.deserialize_logistic_regression(model_dict)
    elif model_dict["meta"] == "linear-regression":
        return reg.deserialize_linear_regressor(model_dict)
    elif model_dict["meta"] == "elasticnet-regression":
        return reg.deserialize_elastic_regressor(model_dict)
    elif model_dict["meta"] == "lasso-regression":
        return reg.deserialize_lasso_regressor(model_dict)
    elif model_dict["meta"] == "ridge-regression":
        return reg.deserialize_ridge_regressor(model_dict)
    elif model_dict["meta"] == "kmeans_clustering":
        return clst.deserialize_kmeans_clustering(model_dict)
    elif model_dict["meta"] == "dbscan_clustering":
        return clst.deserialize_dbscan_clustering(model_dict)
    elif model_dict["meta"] in ["lgbm_multiclass", "rf_multiclass"]:
        return clf.deserialize_lgbm_classifier(model_dict)
    elif model_dict["meta"] in ["lgbm_binary", "rf_binary"]:
        return clf.deserialize_lgbm_binary(model_dict)
    elif model_dict["meta"] in ["lgbm_regressor", "rf_regressor"]:
        return reg.deserialize_lgbm_regressor(model_dict)
    elif model_dict["meta"] == "iforest_anomaly":
        return clst.deserialize_iforest(model_dict)
    else:
        raise ModellNotSupported("Model type not supported or corrupt JSON file")

def from_dict(model_dict):
    return deserialize_model(model_dict)


def from_json(model_name):
    with open(model_name, "r") as model_json:
        model_dict = json.load(model_json)
        return deserialize_model(model_dict)

class ModellNotSupported(Exception):
    pass
