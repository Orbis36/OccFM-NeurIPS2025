import torch

from collections import namedtuple
from .worldmodels import build_wm
from .compressors import build_compressor


def build_network(model_cfg, loss_cfg=None, cache_mode=None):

    if 'TRANSITION_MODEL_CONFIG' in model_cfg.keys():
        model = build_wm(model_cfg=model_cfg, loss_cfg=loss_cfg, cache_mode=cache_mode)
    else:
        model = build_compressor(model_cfg=model_cfg, loss_cfg=loss_cfg)
    return model

def load_data_to_gpu(batch_dict, device=None):
    for key, val in batch_dict.items():
        if key in ['trajectory', 'semantic_occ', 'x_sampled']:
            batch_dict[key] = torch.from_numpy(val).cuda()
        else:
            continue
    return batch_dict


def model_fn_decorator(device):
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict, device)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if model.training:
            if hasattr(model, 'update_global_step'):
                model.update_global_step()
            else:
                model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
