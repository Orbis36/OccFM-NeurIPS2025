import torch
import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from easydict import EasyDict
from torch.optim.swa_utils import AveragedModel

def split_weight(all_parameters):
    params_decay = []
    params_no_decay = []
    for name, param in all_parameters:
        if name.split('.')[0] != 'loss':  # Remove discriminator
            if "ln" in name.lower() or "norm" in name.lower():
                params_no_decay.append(param)
            else:
                params_decay.append(param)
    return params_decay, params_no_decay

def build_optimizer(optimizer_cfg, model, world_size, freeze_compressor=False):

    lr = optimizer_cfg['BASE_LR'] * optimizer_cfg['BATCH_SIZE_PER_GPU'] * world_size

    if freeze_compressor:
        para_need_grad = []
        for name, param in model.named_parameters():
            if name.startswith('transition_model'):
                para_need_grad.append((name, param))
    else:
        para_need_grad = model.named_parameters()

    params_decay, params_no_decay = split_weight(para_need_grad)

    if optimizer_cfg['NAME'] == 'ADAM':
        opt = torch.optim.AdamW([
            {"params": filter(lambda p: p.requires_grad, params_decay), "weight_decay": 0.01},
            {"params": filter(lambda p: p.requires_grad, params_no_decay), "weight_decay": 0}
            # LayerNorm no Weight Decay
        ], lr=lr)
    else:
        raise NotImplementedError
    return opt

def build_scheduler(optimizer, scheduler_cfg, training_length_ep, last_epoch=-1):

    warmup_steps = scheduler_cfg['WARMUP_STEPS']
    min_lr_ratio = scheduler_cfg['MIN_LR_RATIO']
    training_length = training_length_ep * scheduler_cfg['NUM_EPOCHS']

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / (training_length - warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    # here epoch is epoch for optimizer
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
    return scheduler

def build_ema(model, ema_decay=0.9999):
    ema_model = AveragedModel(model, avg_fn=torch.optim.swa_utils.get_ema_avg_fn(ema_decay), use_buffers = True)
    return ema_model

