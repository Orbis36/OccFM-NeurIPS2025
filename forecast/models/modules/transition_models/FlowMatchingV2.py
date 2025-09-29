import torch.nn as nn
import torch
from functools import partial
from einops import rearrange, repeat

from forecast.models.modules.base.others import EinopsToAndFrom, PreNorm
from forecast.models.modules.base.attn_base import Attention
from forecast.models.modules.base.res_layersbase import Residual, ResnetBlock
from forecast.models.modules.base.dit import DiTBlock
from forecast.models.modules.base.postion_embed import (RelativePositionBias, get_2d_sincos_pos_embed, timestep_embedding,
                                                        get_1d_sincos_temp_embed, get_fourier_embeds_from_coordinates)

class FLOW_MATCHING_DOWN_X4_DiT(nn.Module):
    def __init__(self, in_channels=3, out_channels=96, traj_fourier_freq=8, model_channels=None, channel_multi=None,
                 input_size=None, trajectory_length=None, init_kernel_size=None, init_3d_conv_channels=None, attn_dim=None,
                 temporal_attn_head=None, spatial_attn_head=None, **kwargs):
        super().__init__()

        self.skip = kwargs.get('skip', False)
        self.input_size = input_size
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.model_channels = model_channels

        time_embed_dim = model_channels
        self.t_embedder = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.time_rel_pos_bias = RelativePositionBias(heads=temporal_attn_head, max_distance=16)
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(in_channels, init_3d_conv_channels, (1, init_kernel_size, init_kernel_size),
                                   padding=(0, init_padding, init_padding))

        # Init 2+1D conv
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c',
                                                    Attention(dim, heads=temporal_attn_head, dim_head=attn_dim))
        self.init_temporal_attn = Residual(PreNorm(init_3d_conv_channels, temporal_attn(init_3d_conv_channels)))


        # Condition dim need to be added here
        block_klass_cond = partial(ResnetBlock, time_emb_dim=time_embed_dim)

        # modules for all layers
        # for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
        dim_list, dim_in = [], init_3d_conv_channels

        for ind, mult in enumerate(channel_multi):
            is_last = ind == len(channel_multi)-1
            dim_out = mult * init_3d_conv_channels
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out), # channel wise feature, no spatial or temporal aggregation
                block_klass_cond(dim_out, dim_out),

                DiTBlock(dim_out, num_heads=spatial_attn_head, cond_size=time_embed_dim),
                DiTBlock(dim_out, num_heads=spatial_attn_head,  cond_size=time_embed_dim),

                nn.Identity(),
                nn.Conv3d(dim_out, dim_out, (1, 4, 4), (1, 2, 2), (0, 1, 1)) if not is_last else nn.Identity() # down-sample
            ]))
            dim_list.append([dim_in, dim_out])
            dim_in = dim_out

        # mid layer
        mid_dim = dim_in
        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=spatial_attn_head))
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        # up-sampling
        for ind, (dim_in, dim_out) in enumerate(reversed(dim_list)):
            is_last = ind == len(dim_list) - 1
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),
                DiTBlock(dim_in, num_heads=spatial_attn_head, cond_size=time_embed_dim),
                DiTBlock(dim_in, num_heads=spatial_attn_head, cond_size=time_embed_dim),
                nn.ConvTranspose3d(dim_in, dim_in, (1, 4, 4), (1, 2, 2), (0, 1, 1)) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            ResnetBlock(dim_in * 2, dim_in),
            nn.Conv3d(dim_in, out_channels, 1)
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.input_size[0] * self.input_size[1], model_channels), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, 12, model_channels), requires_grad=False)

        self.traj_length = trajectory_length
        self.traj_fourier_freq = traj_fourier_freq
        self.traj_encoder = nn.Sequential(
            nn.Linear(trajectory_length * self.traj_fourier_freq * 4, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pos_embed.shape[-2] ** 0.5) + 1)
        pos_embed = pos_embed[:self.pos_embed.shape[-2]]
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder[2].weight, std=0.02)


    def encode_pose(self, traj):
        batch_size = traj.shape[0]
        traj = rearrange(traj, 'b f c -> (b f) c')
        traj_feat = get_fourier_embeds_from_coordinates(self.traj_fourier_freq, traj)
        traj_feat = rearrange(traj_feat, '(b f) c -> b (f c)', b=batch_size) # per batch
        traj_feat = self.traj_encoder(traj_feat)
        return traj_feat


    def forward_single(self, x, timesteps=None, trajectory=None, **kwargs):


        # init 2+1D embedding
        x = rearrange(x, 'b f c h w -> b c f h w')
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        # Only conv along h&w
        x = self.init_conv(x)
        x_temporal = self.init_temporal_attn(x, pos_bias=time_rel_pos_bias)
        x = x + x_temporal

        r = x.clone()

        # Time embedding
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.t_embedder(t_emb)

        if trajectory is not None:
            trajectory = trajectory[:, :self.traj_length, :]
            traj_feat = self.encode_pose(trajectory)
            emb = traj_feat + emb

        h = []
        timestep_spatial = repeat(emb, 'n d -> (n c) d', c=self.temp_embed.shape[1])
        temp_repeat_time = self.pos_embed.shape[1]
        height_img = self.input_size[0]

        # UNet backbone for denoising
        for idx, (block1, block2, spatial_attn, temporal_attn, identity, downsample) in enumerate(self.downs):
            # Size will decrease with down-sample
            timestep_temp = repeat(emb, 'n d -> (n c) d', c=temp_repeat_time)

            x = block1(x, emb)
            x = block2(x, emb)
            x = rearrange(x, 'b c f h w -> (b f) (h w) c')
            x = x + self.pos_embed if idx == 0 else x
            x = spatial_attn(x, timestep_spatial)

            x = rearrange(x, '(b f) t c -> (b t) f c', b=r.shape[0])
            x = x + self.temp_embed if idx == 0 else x
            x = temporal_attn(x, timestep_temp)
            x = rearrange(x, '(b t) f c -> b c f t', b=r.shape[0])
            x = rearrange(x, 'b c f (h w) -> b c f h w', h=height_img, w=height_img)
            h.append(x)

            x = downsample(x)
            height_img = x.shape[-2]
            temp_repeat_time = x.shape[-1] * x.shape[-2]

        x = self.mid_block1(x, emb)
        x = self.mid_spatial_attn(x)
        x = self.mid_temporal_attn(x, pos_bias=time_rel_pos_bias)
        x = self.mid_block2(x, emb)

        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            timestep_temp = repeat(emb, 'n d -> (n c) d', c=temp_repeat_time)

            x = torch.cat((x, h.pop()), dim=1)
            height_img = x.shape[-2]
            x = block1(x, emb)
            x = block2(x, emb)

            x = rearrange(x, 'b c f h w -> (b f) (h w) c')
            x = spatial_attn(x, timestep_spatial)
            x = rearrange(x, '(b f) t c -> (b t) f c', b=r.shape[0])

            x = temporal_attn(x, timestep_temp)

            x = rearrange(x, '(b t) f c -> b t f c', b=r.shape[0])
            x = rearrange(x, 'b (h w) f c -> b h w f c', h=height_img)
            x = rearrange(x, 'b h w f c -> b c f h w')

            x = upsample(x)

            temp_repeat_time = x.shape[-1] * x.shape[-2]

        x = self.final_conv(torch.cat((x, r), dim=1))
        x = rearrange(x, 'b c f h w -> b f c h w')

        return x


    def forward(self, batch_dict):

        x = batch_dict['noised_sequence']
        timesteps = batch_dict['timesteps']
        trajectory = batch_dict['trajectory']

        predicted_latent = self.forward_single(x, timesteps, trajectory)
        batch_dict['predicted_latent'] = predicted_latent

        return batch_dict