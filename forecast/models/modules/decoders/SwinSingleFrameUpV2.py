import torch
import torch.nn as nn

from ..base.swin_transformerbase import BasicLayer
from ..base.res_layersbase import Residual_conv
from ..base.attn_base import AttnBlock, Normalize
from ..base.others import Upsample, nonlinearity
from ..base.temporal_base import TemporalBlock

from einops import rearrange

class SwinSingleFrameUpV2(nn.Module):
    def __init__(self, input_channel=3, patch_embed_dim=96, window_size=5, heads=8,
                 out_dim=None, depth=None, mlp_ratio=4, qkv_bias=True, attn_drop_rate=0, drop_rate=0, use_checkpoint=False,
                 upsample='conv', patched_size=50,  **kwargs):
        super().__init__()

        self.skip = kwargs.get('skip', False)

        self.depth = len(depth)
        self.net_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
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

            if upsample[i] == 'interp':
                self.upsample_blocks.append(Upsample(patch_embed_dim))
            elif upsample[i] == 'patch_split':
                raise NotImplementedError
            else:
                raise NotImplementedError

            self.temporal_block.append(TemporalBlock(patch_embed_dim, 6)) if (self.use_temporal and i==0) \
                else self.temporal_block.append(nn.Identity())
            patched_size = patched_size * 2

        self.mid = nn.ModuleList()
        self.mid.append(Residual_conv(patch_embed_dim, patch_embed_dim))
        self.mid.append(AttnBlock(patch_embed_dim))
        self.mid.append(Residual_conv(patch_embed_dim, patch_embed_dim))

        self.conv_in = torch.nn.Conv2d(input_channel, patch_embed_dim, kernel_size=3, stride=1, padding=1)
        self.norm_out = Normalize(out_dim)

        self.conv_out = torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, data_dict):

        if self.skip:
            return data_dict

        sampled_features = data_dict['sampled_features']
        compressed_map = self.conv_in(sampled_features)

        compressed_map = self.mid[0](compressed_map)
        compressed_map = self.mid[1](compressed_map)
        compressed_map = self.mid[2](compressed_map)
        compressed_map = rearrange(compressed_map, 'b c h w -> b (h w) c')

        for i in range(self.depth):
            swin_block, res_block,= self.net_blocks[i*2: i*2 +2]
            upsample = self.upsample_blocks[i]

            compressed_map = swin_block(compressed_map)
            compressed_map = rearrange(compressed_map, 'b (h w) c -> b c h w', h=int(compressed_map.shape[1] ** 0.5))

            compressed_map = res_block(compressed_map)

            compressed_map = self.temporal_block[i](compressed_map)
            compressed_map = upsample(compressed_map)


            if i != self.depth - 1:
                compressed_map = rearrange(compressed_map, 'b h w c-> b (h w) c')

        compressed_map = rearrange(compressed_map, 'b h w c-> b c h w')
        compressed_map = self.norm_out(compressed_map)
        compressed_map = nonlinearity(compressed_map)
        compressed_map = self.conv_out(compressed_map)

        data_dict['decoded_map'] = compressed_map
        return data_dict



