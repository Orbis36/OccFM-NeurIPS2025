import copy

import yaml
from easydict import EasyDict

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        new_config = yaml.safe_load(f)
        merge_new_config(config=config, new_config=new_config)

    MODEL = EasyDict()
    for model_module_name in ['COMPRESSOR_CONFIG', 'TRANSITION_MODEL_CONFIG', 'PLANNER_CONFIG']:
        if model_module_name in config.keys():
            MODEL[model_module_name] = copy.deepcopy(config[model_module_name])
            del config[model_module_name]
        config['MODEL'] = MODEL
    config['MODEL']['NAME'] = config['NAME']
    return config