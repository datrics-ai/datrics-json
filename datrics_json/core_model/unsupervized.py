import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.tree import ExtraTreeRegressor
from . import regression as reg


def deserialize_kmeans_clustering(model_dict):
    model = KMeans(model_dict["params"])

    model.cluster_centers_ = np.array(model_dict["cluster_centers_"])
    model.labels_ = np.array(model_dict["labels_"])
    model.inertia_ = model_dict["inertia_"]
    model.n_features_in_ = model_dict["n_features_in_"]
    model.n_iter_ = model_dict["n_iter_"]
    model._n_threads = model_dict["_n_threads"]
    model._tol = model_dict["_tol"]

    return model


def deserialize_dbscan_clustering(model_dict):
    model = DBSCAN(**model_dict["params"])
    model.components_ = np.array(model_dict["components_"])
    model.labels_ = np.array(model_dict["labels_"])
    model.core_sample_indices_ = model_dict["core_sample_indices_"]
    model.n_features_in_ = model_dict["n_features_in_"]
    model._estimator_type = model_dict["_estimator_type"]

    return model


def deserialize_iforest(model_dict):
    model = IsolationForest(**model_dict["params"])

    for param in list(model_dict.keys())[4:-1]:
        setattr(model, param, model_dict[param])

    model.base_estimator_ = ExtraTreeRegressor(**model_dict["base_estimator_"])

    estimators_features_ = list(map(lambda x: np.array(x), model_dict["estimators_features_"]))
    model.estimators_features_ = estimators_features_

    _seeds = np.array(model_dict["_seeds"])
    model._seeds = _seeds

    new_estimators = []
    for est_dict in model_dict["estimators_"]:
        est = ExtraTreeRegressor(**est_dict["params"])
        for param in list(est_dict.keys())[1:-1]:
            setattr(est, param, est_dict[param])
        est.tree_ = reg.deserialize_tree(
            est_dict["tree_"],
            est_dict["n_features_"],
            est.n_classes_[0],
            est_dict["n_outputs_"],
        )
        new_estimators.append(est)
    model.estimators_ = new_estimators

    return model