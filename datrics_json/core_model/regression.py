import lightgbm as lgbm
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

class LGBM_Regression_Booster:
    def __init__(self, booster=None, boosting = 'lgbm'):
        self.booster = lgbm.Booster(model_str=booster)
        self.boosting = boosting

    def predict(self, X):
        return self.booster.predict(X)

def deserialize_linear_regressor(model_dict):
    model = LinearRegression(model_dict["params"])

    model.coef_ = np.array(model_dict["coef_"])
    model.intercept_ = np.array(model_dict["intercept_"])

    return model

def deserialize_lasso_regressor(model_dict):
    model = Lasso(model_dict["params"])
    model.coef_ = np.array(model_dict["coef_"])
    if isinstance(model_dict["n_iter_"], list):
        model.n_iter_ = np.array(model_dict["n_iter_"])
    else:
        model.n_iter_ = int(model_dict["n_iter_"])
    if isinstance(model_dict["intercept_"], list):
        model.intercept_ = np.array(model_dict["intercept_"])
    else:
        model.intercept_ = float(model_dict["intercept_"])

    return model

def deserialize_elastic_regressor(model_dict):
    model = ElasticNet(model_dict["params"])
    model.coef_ = np.array(model_dict["coef_"])
    model.alpha = np.array(model_dict["alpha"]).astype(np.float)
    if isinstance(model_dict["n_iter_"], list):
        model.n_iter_ = np.array(model_dict["n_iter_"])
    else:
        model.n_iter_ = int(model_dict["n_iter_"])

    if isinstance(model_dict["intercept_"], list):
        model.intercept_ = np.array(model_dict["intercept_"])
    else:
        model.intercept_ = float(model_dict["intercept_"])

    return model

def deserialize_ridge_regressor(model_dict):
    model = Ridge(model_dict["params"])
    model.coef_ = np.array(model_dict["coef_"])
    if "n_iter_" in model_dict:
        model.n_iter_ = np.array(model_dict["n_iter_"])
    if isinstance(model_dict["intercept_"], list):
        model.intercept_ = np.array(model_dict["intercept_"])
    else:
        model.intercept_ = float(model_dict["intercept_"])

    return model


def deserialize_lgbm_regressor(model_dict):
    model = LGBM_Regression_Booster(model_dict["booster"])
    return model
