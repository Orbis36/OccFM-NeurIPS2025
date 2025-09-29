import argparse
import glob
import os
import shutil
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch

from os.path import isfile

from forecast.utils import common_utils
from easydict import EasyDict
from pathlib import Path
from forecast.config import cfg_from_yaml_file
from forecast.datasets import build_dataloader, reset_batch_size
from forecast.models import build_network, model_fn_decorator

from train_utils.optimization import build_optimizer, build_scheduler, build_ema
from train_utils.train_utils import train_model
from common_utils.display_utils import setup_loggers, show_eval
from rich.live import Live
from tools.test import val_model

from torch.optim.swa_utils import AveragedModel

from rich.console import Console
from cache_vae import cache_model

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=6, help='number of workers for dataloader')
    parser.add_argument('--local-rank', type=int, default=None, help='local rank for distributed training')

    # recover training only, train from last ckpt, if have finished, just eval val loss
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    # for pretrain weight load, train from scratch
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    parser.add_argument('--cache_mode', action='store_true', default=False, help='')
    parser.add_argument('--amp', action='store_true', default=False, help='')
    parser.add_argument('--skip_opti', action='store_true', default=False, help='')
    parser.add_argument('--use_ema', action='store_true', default=False, help='')
    parser.add_argument('--mu_sigma_cache', action='store_true', default=False, help='')

    parser.add_argument('--save_path', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--eval_mode', action='store_true', default=False, help='')

    cfg = EasyDict()
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.LOCAL_RANK = 0

    return parser.parse_args(), cfg

if __name__ == '__main__':

    args, cfg = parse_config()

    if args.fix_random_seed:
        common_utils.set_random_seed(1000)

    recover_training = False

    if getattr(args, 'ckpt', None) is not None and isfile(args.ckpt):
        ori_run_name = '/'.join(args.ckpt.rstrip('/').split('/')[:-2])
        ori_yaml_path = glob.glob(ori_run_name + '/*.yaml')
        assert len(ori_yaml_path) == 1 and ori_yaml_path[0].split('/')[-1] == args.cfg_file.split('/')[-1], "YAML confliction"
        args.cfg_file = ori_yaml_path[0]
        recover_training = True

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if not recover_training:
        output_dir = cfg.ROOT_DIR / 'logs' / cfg.TAG / args.extra_tag
        ckpt_dir = output_dir / 'ckpt'
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.cfg_file, output_dir)
        output_dir = str(output_dir)
    else:
        output_dir = ori_run_name

    print("----------- Create dataloader & network & optimizer -----------")

    batch_size = args.batch_size if args.batch_size is not None \
        else cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU * 1

    local_rank = int(os.environ["LOCAL_RANK"])  # GPU id on this node
    rank = int(os.environ["RANK"])  # global rank
    world_size = int(os.environ["WORLD_SIZE"])
    assert local_rank == rank, "Current only support 1 node"

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    is_main_process = (rank == 0)

    # cfg.CACHE_MODE to skip encoder
    model = build_network(model_cfg=cfg.MODEL, loss_cfg=cfg.LOSS, cache_mode=cfg.CACHE_MODE).to(rank)
    ema_model = build_ema(model).to(rank) if args.use_ema else None

    train_set, train_loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, batch_size=batch_size, num_workers=args.workers,
        cache_mode=cfg.CACHE_MODE, rank=rank, world_size=world_size
    )

    val_set, val_loader = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, batch_size=batch_size, num_workers=args.workers,
        cache_mode=cfg.CACHE_MODE, training=False, rank=rank, world_size=world_size
    )

    optimizer = build_optimizer(cfg.OPTIMIZATION, model, world_size)
    progress, console = setup_loggers()

    if recover_training:
        model_status = model.recover_training(args.ckpt)
        scheduler = build_scheduler(optimizer, cfg.OPTIMIZATION, training_length_ep=len(train_loader), last_epoch=-1)
        # For compatible with weight from old repo
        # TODO: remove this
        if not args.skip_opti:
            optimizer.load_state_dict(model_status['optimizer_states'][0])
            scheduler.load_state_dict(model_status['lr_scheduler'])
        if ema_model is not None:
            ema_model.load_state_dict(model_status['ema_model'])

        if cfg.OPTIMIZATION.NUM_EPOCHS > model_status['epoch'] and not args.eval_mode:
            train_model(model, optimizer, train_loader, val_loader, scheduler, console=console, progress=progress, is_main_process=is_main_process,
                        ckpt_path=output_dir + '/ckpt/', start_epoch=model_status['epoch'], optim_cfg=cfg.OPTIMIZATION, rank=rank, ema_model=ema_model,
                        eval_interval=cfg.EVAL_INTERVAL, model_func=model_fn_decorator(rank), loss_monitor=cfg.LOSS_MONITOR, use_amp=args.amp)
        else:
            if is_main_process:
                console.print(
                    "[bold magenta]✔️ All training epochs completed. The model is fully trained and ready for evaluation or deployment.[/bold magenta]")

    else:
        # If from scratch, recover data compressor parameters, then build ema model
        _ = model.recover_compressor(cfg.COMPRESSOR_CONFIG.PRETRAIN_WEIGHT) if hasattr(model, 'transition_model') else None
        scheduler = build_scheduler(optimizer, cfg.OPTIMIZATION, training_length_ep=len(train_loader))

        train_model(model, optimizer, train_loader, val_loader, scheduler, console=console, progress=progress, rank=rank,
        use_amp = args.amp, ckpt_path=output_dir + '/ckpt/', start_epoch=0, optim_cfg=cfg.OPTIMIZATION, is_main_process=is_main_process,
        eval_interval=cfg.EVAL_INTERVAL, model_func=model_fn_decorator(rank), loss_monitor=cfg.LOSS_MONITOR, ema_model=ema_model)

    model = ema_model if ema_model is not None else model

    # cache and fps
    with Live(console=console, refresh_per_second=2, transient=True) as live:

        model = DDP(model, device_ids=[rank]) if recover_training else None
        test_cfm = hasattr(model.module.module, 'transition_model') if isinstance(model.module, AveragedModel) else \
            hasattr(model.module, 'transition_model')

        train_loader = reset_batch_size(train_loader, 1, rank=rank, world_size=world_size, training=True)
        val_loader = reset_batch_size(val_loader, 1, rank=rank, world_size=world_size)

        val_avg_loss = val_model(model, val_loader, model_fn_decorator(rank), progress, live, rank=rank,
                                 test_cfm=test_cfm, use_amp=args.amp, eval_iou=True, eval_fps=True,
                                 is_main_process=is_main_process)

        if is_main_process:
            show_eval(val_avg_loss, console)

        cache_mu_sigma = args.mu_sigma_cache or cfg.get("MU_SIGMA_CACHE", False)

        # TODO: fix the duplicated parameters
        # args.cache_mode used to control cache latent or not, used in VAE inference
        # cfg.CACHE_MODE used to skip the compressor in latent CFM when inference
        if args.cache_mode:
            cache_model(model, [train_loader, val_loader], model_fn_decorator(rank), progress, live=live, console=console,
                        cache_mode=args.cache_mode, mu_sigma_cache=cache_mu_sigma, save_path=args.save_path, is_main_process=is_main_process)

    dist.barrier(device_ids=[rank])
    dist.destroy_process_group()