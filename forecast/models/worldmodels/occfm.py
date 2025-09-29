import torch
import torch.nn as nn
import numpy as np
from easydict import EasyDict
import torch.nn.functional as F
from einops import reduce, rearrange

from forecast.models.model_template import ModelTemplate
from forecast.utils.common_utils import merge_dicts, cuda_timer, cpu_timer


class OccFM(ModelTemplate):
    def __init__(self, model_cfg, loss_cfg, **kwargs):
        super().__init__(model_cfg=merge_dicts(model_cfg))

        skip_list = self.compressor_topology if kwargs.get('cache_mode', None) is not None else []
        self.module_list = self.build_model(self.world_model_topology, skip_list)
        self.uncond_p = loss_cfg['UNCOND_P']
        self.rescale_factor = loss_cfg['RESCALE_FACTOR']

        self.sample_step = loss_cfg['SAMPLE_STEP']
        self.alpha = loss_cfg['ALPHA_STEP']
        self.unconditional_guidance_scale = loss_cfg['UNCOND_SCALE']
        self.time_scalar = 1000

    @staticmethod
    def get_sigmoid_time_sample(bs, device):
        return torch.sigmoid(torch.randn(bs, 1, 1, 1, 1)).to(device)

    def training_step(self, condition_clip, future_clip, batch_size, batch_dict):
        mask_cond = torch.rand(condition_clip.shape[0]) > self.uncond_p
        mask_cond = mask_cond[:, None, None, None, None].int().to(condition_clip.device)
        condition_clip_cfg = condition_clip * mask_cond
        seq_length = condition_clip.shape[1]

        t = self.get_sigmoid_time_sample(batch_size, condition_clip.device)
        noise_z0 = torch.randn_like(future_clip)
        noised_future = t * future_clip + (1. - t) * noise_z0

        net_input = torch.concat((condition_clip_cfg, noised_future), dim=1)

        batch_dict['noised_sequence'] = net_input
        batch_dict['timesteps'] = t.squeeze() * self.time_scalar

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        future_seq = batch_dict['predicted_latent'][:, seq_length:, ...]
        target = future_clip - noise_z0
        return future_seq, target

    def sample(self, batch_dict, batch_size):
        """
        input: full sequence with pure noise future
        traj: trajectory for each batch
        """
        ## t shifting from stable diffusion 3
        input = batch_dict['x_sampled']

        timesteps = torch.linspace(0, 1, self.sample_step + 1)
        t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0)

        traj = batch_dict['trajectory']
        batch_dict['trajectory'] = torch.cat([traj] * 2) if traj is not None else traj

        # Solve ODE
        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            t = torch.tensor([t_curr * self.time_scalar]).unsqueeze(0).repeat(input.shape[0], 1).cuda().reshape(1, -1)

            # classifier-free guidance
            cond_seq, future_seq = torch.chunk(input, 2, dim=1)
            uncond_seq = torch.concat((torch.zeros_like(cond_seq), future_seq), dim=1)
            cfg_input = torch.concat((uncond_seq, input), dim=0)
            t = torch.cat([t] * 2, dim=1).reshape(-1, )

            batch_dict['noised_sequence'] = cfg_input
            batch_dict['timesteps'] = t.squeeze()

            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
            v = batch_dict['predicted_latent']

            uncond_v, cond_v = torch.chunk(v, 2, dim=0)
            v = uncond_v + self.unconditional_guidance_scale * (cond_v - uncond_v)
            # only v for last 6 frames are useful
            _, v_future = torch.chunk(v, 2, 1)

            # The noised future
            _, future_frames_with_cfg = torch.chunk(cfg_input, 2, 1)
            denoised_future_frames = future_frames_with_cfg.clone()[batch_size:, ...] + step * v_future
            input = torch.concat((cond_seq, denoised_future_frames), dim=1)

        _, output = torch.chunk(input, 2, dim=1)
        output /= self.rescale_factor

        return output

    @cuda_timer
    def cfm_eval(self, future_clip, condition_clip, batch_dict, batch_size):

        start_future_latent = torch.randn_like(future_clip)
        batch_dict['x_sampled'] = torch.concat((condition_clip, start_future_latent), dim=1)

        assert batch_size == 1, "cfm eval time = 1 only now"
        batch_dict['sampled_features'] = self.sample(batch_dict, batch_size).squeeze(0)

        # only use decoder in sampling
        self.decoder.skip = False
        batch_dict = self.decoder(batch_dict)
        self.decoder.skip = True

        decoded_map = batch_dict['decoded_map']
        template = self.embedding.class_embeds.weight.T.unsqueeze(0).detach()
        decoded_map = rearrange(decoded_map, 'b (d c) h w -> b h w d c', d=self.embedding.height_num,
                                c=self.embedding.cate)
        similarity = torch.matmul(decoded_map, template)

        pred_occ = similarity.permute(0, 4, 1, 2, 3).softmax(dim=1)
        pred_occ = torch.argmax(pred_occ.softmax(dim=1), 1).unsqueeze(1)

        loss, tb_dict, disp_dict = self.get_training_loss(batch_dict['sampled_features'].unsqueeze(0), future_clip)
        disp_dict['pred_occ'] = pred_occ.squeeze(1).unsqueeze(0)
        return (loss, tb_dict, disp_dict)


    def forward(self, batch_dict):

        x_sampled = batch_dict['x_sampled']
        batch_size = len(batch_dict['paths'])

        x_sampled = self.rescale_factor * x_sampled
        condition_clip, future_clip = torch.chunk(x_sampled, 2, dim=1)

        if batch_dict.get('cfm_eval', False):
            output, time = self.cfm_eval(future_clip, condition_clip, batch_dict, batch_size)
            loss, tb_dict, disp_dict = output
            disp_dict["time"] = time

        else:
            future_seq, target = self.training_step(condition_clip, future_clip, batch_size, batch_dict)
            loss, tb_dict, disp_dict = self.get_training_loss(future_seq, target)

        return loss, tb_dict, disp_dict

    def nn_forward(self, x):
        pass

    def get_training_loss(self, future_seq, target):
        tb_dict, disp_dict = {}, {}

        train_loss = F.mse_loss(future_seq, target, reduction='none')
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        loss = train_loss.mean()

        tb_dict['mse_loss'] = loss
        tb_dict['loss'] = loss

        disp_dict['future_seq'] = future_seq

        return tb_dict, tb_dict, disp_dict