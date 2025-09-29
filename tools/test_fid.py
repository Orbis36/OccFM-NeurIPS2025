import argparse
import glob
from os.path import isfile

from pathlib import Path

import torch
from easydict import EasyDict

from forecast.utils import common_utils
from forecast.config import cfg_from_yaml_file
from forecast.models import build_network
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import numpy as np
import re
import scipy.linalg as sla

from forecast.datasets.inception_metric_dataset import FVDEval

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--eval_model', type=str, default=None, help='checkpoint to start from')
    # for pretrain weight load, train from scratch
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')

    parser.add_argument('--amp', action='store_true', default=False, help='')
    parser.add_argument('--skip_opti', action='store_true', default=False, help='')
    cfg = EasyDict()
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.LOCAL_RANK = 0

    return parser.parse_args(), cfg


if __name__ == '__main__':
    args, cfg = parse_config()

    if args.fix_random_seed:
        common_utils.set_random_seed(1000)

    assert isfile(args.ckpt), "should have a weight for fid eval"

    ori_run_name = '/'.join(args.ckpt.rstrip('/').split('/')[:-2])
    ori_yaml_path = glob.glob(ori_run_name + '/*.yaml')
    assert len(ori_yaml_path) == 1 and ori_yaml_path[0].split('/')[-1] == args.cfg_file.split('/')[-1], "YAML confliction"
    args.cfg_file = ori_yaml_path[0]

    output_dir = ori_run_name

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    model = build_network(model_cfg=cfg.MODEL, loss_cfg=cfg.LOSS, cache_mode=cfg.CACHE_MODE).cuda()
    model.eval()
    model_status = model.recover_training(args.ckpt)

    eval_dataset = FVDEval(args.eval_model)
    dataloader = DataLoader(eval_dataset, batch_size=1, num_workers=2, shuffle=False, drop_last=False,
                            collate_fn=None)
    all_pred = []
    all_gt = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc="Reconstructing Batches"):
            gt, pred = batch[0].cuda().squeeze(0), batch[1].cuda().squeeze(0)

            indices = torch.randperm(gt.size(1))  # 生成一个随机排列的索引
            pred = gt[:, indices, ...]

            batch_dict_pred, _ = model.nn_forward({'semantic_occ': pred})
            pred_latent = batch_dict_pred['sampled_features']
            pooled_pred = F.adaptive_avg_pool2d(pred_latent, (5, 5)).reshape(6, -1).flatten()

            batch_dict_gt, _ = model.nn_forward({'semantic_occ': gt})
            gt_latent = batch_dict_gt['sampled_features']
            pooled_gt = F.adaptive_avg_pool2d(gt_latent, (5, 5)).reshape(6, -1).flatten()

            all_gt.append(pooled_gt)
            all_pred.append(pooled_pred)

    all_pred = torch.stack(all_pred).cpu().numpy()
    all_gt = torch.stack(all_gt).cpu().numpy()

    mu_R, mu_G = all_gt.mean(0), all_pred.mean(0)
    Sigma_R = np.cov(all_gt, rowvar=False)
    Sigma_G = np.cov(all_pred, rowvar=False)

    covmean, _ = sla.sqrtm(Sigma_R @ Sigma_G, disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real

    cov = np.trace(Sigma_R + Sigma_G - 2 * covmean)
    mean = np.sum((mu_R - mu_G) ** 2)

    fid = np.sum((mu_R - mu_G) ** 2) + np.trace(Sigma_R + Sigma_G - 2 * covmean)
    print(fid)