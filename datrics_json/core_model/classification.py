import lightgbm as lgbm
import numpy as np
from sklearn.linear_model import LogisticRegression

class LGBM_Classifier_Booster:
    def __init__(self, classes=None, booster=None, boosting = 'lgbm'):
        self.classes = np.array(classes)
        self.booster = lgbm.Booster(model_str=booster)
        self.objective = 'multiclass'
        self.boosting = boosting

    def predict(self, X):
        return self.classes[np.argmax(self.booster.predict(X), axis=1)]

    def predict_proba(self, X):
        return self.booster.predict(X)

class LGBM_Binary_Booster:
    def __init__(self, classes=None, booster=None, boosting = 'lgbm'):
        self.classes = np.array(classes)
        self.booster = lgbm.Booster(model_str=booster)
        self.objective = 'binary'
        self.boosting = boosting

    def pserializationredict(self, X):
        probas = np.array(list(map(lambda x: [1 - x, x], self.booster.predict(X).tolist())))
        return self.classes[np.argmax(probas, axis=1)]

    def predict_proba(self, X):
        return self.booster.predict(X)


def deserialize_logistic_regression(model_dict):
    model = LogisticRegression(model_dict["params"])

    model.classes_ = np.array(model_dict["classes_"])
    model.coef_ = np.array(model_dict["coef_"])
    model.intercept_ = np.array(model_dict["intercept_"])
    model.n_iter_ = np.array(model_dict["intercept_"])

    return model


def deserialize_lgbm_classifier(model_dict):
    model = LGBM_Classifier_Booster(classes=model_dict["classes"],
                                    booster=model_dict["booster"],
                                    boosting=model_dict["boosting"])
    return model


def deserialize_lgbm_binary(model_dict):
    model = LGBM_Binary_Booster(classes=model_dict["classes"],
                                booster=model_dict["booster"],
                                boosting=model_dict["boosting"])
    return model
