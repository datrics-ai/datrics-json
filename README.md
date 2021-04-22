# Descripton
Open source library for the **Datrics** models deserialization

## Initial source
The solution is based on https://github.com/mlrequest/sklearn-json library


## Getting Started

datrics-json makes importing the models imlemented in the **Datrics AI platform** from their JSON representation

### Install
```
pip install datrics-json
```
### Example Usage

```python
import datrics_json as datjson

model_dict = datjson.from_json(file_name)
deserialized_model = list(model_dict.get('trained_models').values())[0]['model']

sample_data = model_dict.get('sample_data')['init']

deserialized_model.predict(sample_data)
```

## Features
sklearn-json requires scikit-learn >= 0.22.2.
LightGBM >= 2.3.1

### Supported scikit-learn Models
 * sklearn.linear_model.LogisticRegression
 * sklearn.ensemble.IsolationForest
 * sklearn.clustering.KMeans
 * sklearn.clustering.DBSCAN
 * sklearn.linear_model.LinearRegression
 * sklearn.linear_model.Ridge
 * sklearn.linear_model.Lasso
 * sklearn.linear_model.ElasticNet

### Supported lightGBM Models
   * lightgbm.LGBMClassifier - binary - Gradient Boosting Trees
   * lightgbm.LGBMClassifier - multiclass - Gradient Boosting Trees
   * lightgbm.LGBMClassifier - binary - Random Forest
   * lightgbm.LGBMClassifier - multiclass - Random Forest
   * lightgbm.LGBMRegressor - Gradient Boosting Trees
   * lightgbm.LGBMRegressor - Random Forest

# Test data
   * [Examples of JSON Datrics models represendation](data)

