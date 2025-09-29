import random
import numpy as np
import torch
from easydict import EasyDict
from itertools import repeat
import collections.abc
import functools
import time

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def merge_dicts(model_cfg):
    merged_dict = EasyDict()
    merged_dict['MODEL'] = EasyDict()

    for module_name in model_cfg.COMPRESSOR_CONFIG.MODEL.keys():
        if module_name.lower() in ['embedding', 'encoder', 'quantization', 'transition_model', 'decoder']:
            merged_dict['MODEL'][module_name] = model_cfg.COMPRESSOR_CONFIG.MODEL[module_name]

    for module_name in model_cfg.TRANSITION_MODEL_CONFIG.MODEL.keys():
        merged_dict['MODEL'][module_name] = model_cfg.TRANSITION_MODEL_CONFIG.MODEL[module_name]

    if model_cfg.get('PLANNER_CONFIG', False):
        merged_dict['MODEL']['PLANNER_CONFIG'] = model_cfg['PLANNER_CONFIG']

    return merged_dict


def remove_module_prefix_from_ddp(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_k] = v
    return new_state_dict

def lowercase_keys(d):
    if isinstance(d, dict):
        return EasyDict({k.lower(): lowercase_keys(v) for k, v in d.items()})
    return d

def worker_init_fn(worker_id, seed=666):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cuda_timer(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        starter.record()

        out = fn(*args, **kwargs)

        ender.record()
        torch.cuda.synchronize()
        elapsed = starter.elapsed_time(ender) / 1000.0
        #elapsed = (end - start) / 1000

        return out, elapsed
    return wrapper

def cpu_timer(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()

        out = fn(*args, **kwargs)

        end = time.time()
        elapsed = end - start

        return out, elapsed
    return wrapper