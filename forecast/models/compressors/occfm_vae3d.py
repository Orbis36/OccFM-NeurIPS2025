import torch
import torch.nn as nn
import time
from forecast.models.model_template import ModelTemplate
from einops import rearrange

from forecast.utils.common_utils import cuda_timer
from forecast.utils.loss_utils import lovasz_softmax

import torch.nn.functional as F

class OccFmVAE3D(ModelTemplate):
    def __init__(self, model_cfg, loss_cfg, **kwargs):
        super().__init__(model_cfg.COMPRESSOR_CONFIG)
        self.input_height = self.model_cfg.EMBEDDING.HEIGHT_NUM
        self.cate = self.model_cfg.EMBEDDING.FEAT_DIM
        self.loss_weight = loss_cfg

        self.module_list = self.build_model(self.compressor_topology, skip_list=[])

    @cuda_timer
    def nn_forward(self, batch_dict):

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        decoded_map = batch_dict['decoded_map']

        decoded_map = decoded_map.unsqueeze(0) if len(decoded_map.shape) == 4 else decoded_map

        template = self.embedding.class_embeds.weight.T.unsqueeze(0).detach()
        decoded_map = rearrange(decoded_map, 'b f (d c) h w -> b f h w d c', d=self.input_height, c=self.cate)
        similarity = torch.matmul(decoded_map, template)
        batch_dict['similarity'] = similarity

        return batch_dict

    def forward(self, batch_dict, **kwargs):
        eval_fps = batch_dict.get('eval_fps', False)
        batch_dict, forward_time = self.nn_forward(batch_dict)
        loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)

        if eval_fps and forward_time is not None:
            disp_dict['time'] = forward_time

        return loss, tb_dict, disp_dict

    def get_training_loss(self, batch_dict):

        pred_occ, gt_occ = batch_dict['similarity'], batch_dict['semantic_occ']
        gt_occ = gt_occ.long()

        gt_occ = rearrange(gt_occ, 'b f h w d-> (b f) h w d')
        pred_occ = rearrange(pred_occ, 'b f h w d c-> (b f) h w d c')

        tb_dict, disp_dict = {}, {}

        rec_loss = F.cross_entropy(pred_occ.permute(0, 4, 1, 2, 3), gt_occ, ignore_index=-100)
        weighted_rec_loss = self.loss_weight['RECON_LOSS_WEIGHT'] * rec_loss
        tb_dict['weighted_rec_loss'] = weighted_rec_loss

        # miou
        pred_occ = pred_occ.permute(0, 4, 1, 2, 3).softmax(dim=1)
        loss = lovasz_softmax(pred_occ, gt_occ)
        weighted_lova_loss = self.loss_weight['LOVASZ_LOSS_WEIGHT'] * loss
        tb_dict['weighted_lova_loss'] = weighted_lova_loss

        # KL divergence
        kl_loss = self.quantization.get_loss()
        weighted_kl_loss = self.loss_weight['KL_DIVERGENCE_WEIGHT'] * kl_loss
        tb_dict['weighted_kl_loss'] = weighted_kl_loss

        loss = weighted_rec_loss + weighted_lova_loss + weighted_kl_loss
        tb_dict['loss'] = loss

        disp_dict['pred_occ'] = torch.argmax(pred_occ.softmax(dim=1), 1).unsqueeze(1)
        disp_dict['gt_occ'] = gt_occ.unsqueeze(1)
        disp_dict['gt_path'] = batch_dict['paths']
        disp_dict['trajectory'] = batch_dict['trajectory']

        if batch_dict['mu'] is not None:
            disp_dict['mu'] = batch_dict['mu']
            disp_dict['sigma'] = batch_dict['sigma']

        if self.quantization.latent_cache:
            disp_dict['x_sampled'] = batch_dict['sampled_features']

        return tb_dict, tb_dict, disp_dict