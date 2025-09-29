import torch
import torch.nn as nn

from ..base.swin_transformerbase import PatchEmbed, PatchMerging, BasicLayer
from ..base.res_layersbase import Residual_conv
from ..base.attn_base import AttnBlock
from ..base.others import Normalize, nonlinearity
from ..base.temporal_base import TemporalBlock

from einops import rearrange

class SwinSingleFrameDownV2(nn.Module):
    def __init__(self, img_size=800, patch_size=4, input_channel=3, patch_embed_dim=96, window_size=5, heads=8, single_stride=False,
                 out_dim=None, depth=None, mlp_ratio=4, qkv_bias=True, attn_drop_rate=0, drop_rate=0, use_checkpoint=False,
                 use_lora=False, rank=128, latent_dim=8, downsample='conv', **kwargs):
        super().__init__()

        self.skip = kwargs.get('skip', False)
        patched_size = img_size // patch_size  # No patch for semantic occ
        self.patch_embed_layer = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=input_channel,
                                            embed_dim=patch_embed_dim)
        num_patches = self.patch_embed_layer.num_patches
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, input_channel))

        self.depth = len(depth)

        self.net_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        self.temporal_block = nn.ModuleList()
        self.use_temporal = kwargs.get('temporal_block', False)

        for i in range(len(depth)):
            self.net_blocks.append(
                BasicLayer(dim=patch_embed_dim, input_resolution=(patched_size, patched_size),
                           depth=depth[i], num_heads=heads, window_size=window_size,
                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                           norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=use_checkpoint)
            )
            self.net_blocks.append(
                Residual_conv(patch_embed_dim, patch_embed_dim)
            )


            if downsample[i] == 'conv':
                self.downsample_blocks.append(
                    nn.Conv2d(patch_embed_dim, patch_embed_dim * 2, kernel_size=3, stride=2, padding=1)
                )
            elif downsample[i] == 'patch_merge':
                self.downsample_blocks.append(
                    PatchMerging(input_resolution=(patched_size, patched_size), dim=patch_embed_dim)
                )
            else:
                raise NotImplementedError

            self.temporal_block.append(TemporalBlock(patch_embed_dim * 2, 6)) if self.use_temporal \
                else self.temporal_block.append(nn.Identity())

            patched_size = patched_size // 2
            patch_embed_dim = patch_embed_dim * 2

        self.conv_reduce = torch.nn.Conv2d(patch_embed_dim, patch_embed_dim // 4, kernel_size=3, stride=1, padding=1)

        self.mid = nn.ModuleList()
        self.mid.append(Residual_conv(patch_embed_dim // 4, patch_embed_dim // 4))
        self.mid.append(AttnBlock(patch_embed_dim // 4))
        self.mid.append(Residual_conv(patch_embed_dim // 4, patch_embed_dim // 4))

        self.norm_out = Normalize(patch_embed_dim // 4)
        self.conv_out = torch.nn.Conv2d(patch_embed_dim // 4, latent_dim * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, data_dict):

        if self.skip:
            return data_dict

        bev_feature = data_dict['bev_features']
        bev_feature = self.patch_embed_layer(bev_feature)
        bev_feature = bev_feature + self.absolute_pos_embed.to(bev_feature.dtype)

        for i in range(self.depth):
            swin_block, res_block,= self.net_blocks[i*2: i*2 +2]
            downsample = self.downsample_blocks[i]

            bev_feature = swin_block(bev_feature)
            bev_feature = rearrange(bev_feature, 'b (h w) c -> b c h w', h=int(bev_feature.shape[1] ** 0.5))

            bev_feature = res_block(bev_feature)

            bev_feature = downsample(bev_feature)
            bev_feature = self.temporal_block[i](bev_feature)

            if i != self.depth - 1:
                bev_feature = rearrange(bev_feature, 'b c h w -> b (h w) c')

        bev_feature = self.conv_reduce(bev_feature)

        bev_feature = self.mid[0](bev_feature)
        bev_feature = rearrange(bev_feature, 'b c h w -> b (h w) c')
        bev_feature = self.mid[1](bev_feature)
        bev_feature = self.mid[2](bev_feature)

        bev_feature = self.norm_out(bev_feature)
        bev_feature = nonlinearity(bev_feature)
        compressed_features = self.conv_out(bev_feature)

        data_dict['compressed_features'] = compressed_features
        return data_dict





