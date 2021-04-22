import pandas as pd

from datrics_json import core_model as datjson
import copy

def set_param(datrics_model, json_model, param, parent_param,  new_param = None):
    if new_param is None:
        new_param = param
    value = json_model.get(param,{})
    if len(value) > 0:
        datrics_model[parent_param] = datrics_model.get(parent_param,{})
        if param in ['sample_data', 'sample_output']:
            value = pd.DataFrame(value)
        datrics_model[parent_param][new_param] = value

def deserealized_classifier(json_model):
    datrics_model = {}

    datrics_model['meta'] = json_model['meta']
    set_param(datrics_model, json_model, "model_init_parameters", 'model_parameters')
    set_param(datrics_model, json_model, "model_fit_parameters", 'model_parameters')
    set_param(datrics_model, json_model, "model_predict_parameters", 'model_parameters')
    set_param(datrics_model, json_model, "additional_parameters", 'model_parameters')
    set_param(datrics_model, json_model, "supported_category_values", 'model_parameters')
    set_param(datrics_model, json_model, "grouping_columns", 'model_parameters')

    set_param(datrics_model, json_model, "required_arguments", 'model_arguments')
    set_param(datrics_model, json_model, "optional_arguments", 'model_arguments')
    set_param(datrics_model, json_model, "required_arguments_types", 'model_arguments')
    set_param(datrics_model, json_model, "optional_arguments_types", 'model_arguments')
    set_param(datrics_model, json_model, "keep_columns", 'model_arguments')
    set_param(datrics_model, json_model, "predictions_quality", 'quality')

    set_param(datrics_model, json_model, "sample_data", 'sample_data', 'input')
    set_param(datrics_model, json_model, "sample_output", 'sample_data', 'output')
    set_param(datrics_model, json_model, "transformed_data_columns", 'model_parameters', 'predictors')

    trained_models = copy.deepcopy(json_model["trained_models"])

    for k in trained_models["model"].keys():
        trained_model = datjson.from_dict(trained_models["model"][k])
        trained_models["model"][k] = trained_model

    datrics_model['trained_models'] = pd.DataFrame.from_dict(trained_models)
    indx = datrics_model['trained_models'].index.astype(json_model.get("trained_models_index", 'float'))
    datrics_model['trained_models'].index = indx

    datrics_model['trained_models'] = datrics_model['trained_models'].to_dict('index')

    return datrics_model
