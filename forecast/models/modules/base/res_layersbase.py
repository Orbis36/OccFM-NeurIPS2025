import torch.nn as nn
from .others import Normalize, nonlinearity, RMSNorm, exists
from einops import rearrange

class Residual_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.norm = Normalize(out_channel)

    def forward(self, latent_features):
        conv1 = self.conv1(latent_features)
        # conv1 = rearrange(conv1, 'b c h w -> b h w c')
        input_dtype = conv1.dtype
        conv1 = self.norm(conv1).to(input_dtype)
        conv1 = nonlinearity(conv1)
        conv2 = self.conv2(conv1)
        if conv2.shape[1] != latent_features.shape[1]:
            return conv2
        return conv2 + latent_features


class ResidualConv3D(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=None, nonlinearity=nn.SiLU()):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.norm = nn.GroupNorm(num_groups=32, num_channels=out_channel)
        self.nonlinearity = nonlinearity

    def forward(self, latent_features):
        x = self.conv1(latent_features)
        x = self.norm(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)

        if x.shape[1] != latent_features.shape[1]:
            return x  # No residual if channels mismatch
        return x + latent_features


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding = (0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, traj_embed_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.traj_mlp =  nn.Sequential(
            nn.SiLU(),
            nn.Linear(traj_embed_dim, dim_out * 2)
        ) if exists(traj_embed_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, traj_embed=None):
        scale_shift = None
        if exists(self.mlp) and time_emb is not None:
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')

            if traj_embed is not None:
                traj_embed = self.traj_mlp(traj_embed)
                traj_embed = rearrange(traj_embed, 'b f c -> b c f 1 1')
                time_emb = time_emb + traj_embed

            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)