import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class VoxelHeightSemEmbed(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()

        feat_dim = model_cfg.FEAT_DIM
        voxel_size = model_cfg.VOXEL_SIZE
        self.height_num = model_cfg.HEIGHT_NUM
        self.cate_dim = model_cfg.NUM_CATE
        self.cate = model_cfg.FEAT_DIM

        self.skip = model_cfg['skip']
        self.class_embeds = nn.Embedding(model_cfg.NUM_CATE, feat_dim)
        self.height_embedding = nn.Parameter(torch.zeros(self.height_num, feat_dim // 2), requires_grad=False)
        self._init_embedding_weights(self.height_num, model_cfg.OCC_RANGE, feat_dim // 2, voxel_size)

        self.pre_pillar_feature = nn.Sequential(
            nn.Conv2d(192, model_cfg.ENCODER_DIM , 3, 1, 1),
        )

    def _init_embedding_weights(self, num_z_features, voxel_range, feat_expand, voxel_size):

        possible_height = (np.arange(num_z_features) * voxel_size[-1]) + voxel_size[-1] / 2
        norm_height = possible_height / (voxel_range[-1] -  voxel_range[-4])

        encodings = []
        freqs = feat_expand // 2
        for freq in range(freqs):
            proj_value = torch.as_tensor(norm_height * freq * np.pi)
            encodings.append(torch.sin(proj_value))
            encodings.append(torch.cos(proj_value))
        encodings_data = torch.vstack(encodings).transpose(1, 0)
        self.height_embedding.data.copy_(encodings_data.float())

    def forward(self, data_dict):

        if self.skip:
            return data_dict

        sem_occ = data_dict["semantic_occ"]
        input = self.class_embeds(sem_occ.long())
        height_range = torch.arange(0, 16, device=input.device).long()

        if len(sem_occ.shape) == 4:
            height_embed = self.height_embedding[height_range][None, None, None, ...]
            height_embed = height_embed.expand(input.shape[0], input.shape[1], input.shape[2], -1, -1)
            input = torch.concat((input, height_embed), dim=-1)
            input = rearrange(input, 'b h w c d -> b (c d) h w')
            input = self.pre_pillar_feature(input)

        elif len(sem_occ.shape) == 5: # video embedding
            frame = data_dict["semantic_occ"].shape[1]
            height_embed = self.height_embedding[height_range][None, None, None, None, ...]
            height_embed = height_embed.expand(input.shape[0], input.shape[1], input.shape[2], input.shape[3], -1, -1)
            input = torch.concat((input, height_embed), dim=-1)
            input = rearrange(input, 'b f h w c d -> (b f) (c d) h w')
            input = self.pre_pillar_feature(input)
            # input = rearrange(input, '(b f) p h w -> b f p h w', f=frame)

        data_dict['bev_features'] = input
        return data_dict