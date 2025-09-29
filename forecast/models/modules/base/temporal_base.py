import torch.nn as nn
import torch

from .attn_base import DiTAttention
from .res_layersbase import ResidualConv3D
from einops import rearrange

class TemporalBlock(nn.Module):
    def __init__(self, input_dim, frame, **kwargs):
        super().__init__()
        self.frame = frame
        self.temporal_attn = DiTAttention(input_dim, num_heads=4, qkv_bias=True, attention_mode='math')
        self.temporal_conv_1 = ResidualConv3D(input_dim, input_dim)
        self.temporal_conv_2 = ResidualConv3D(input_dim, input_dim)

    def forward(self, x):
        # x: (B*F, H, W, C)
        # Step 1: reshape to (B, F, H, W, C)
        x = rearrange(x, '(b f) c h w -> b c f h w', f=self.frame)

        # Step 2: 1st residual 3D conv
        x = self.temporal_conv_1(x)

        # Step 3: attention expects (B, F, HW, C)
        x_attn = rearrange(x, 'b c f h w -> (b h w) f c')
        x_attn = self.temporal_attn(x_attn)

        # Step 4: back to (B, C, F, H, W)
        x = rearrange(x_attn, '(b h w) f c -> b c f h w', b=x.shape[0], h=x.shape[-2], w=x.shape[-1])

        # Step 5: 2nd residual 3D conv
        x = self.temporal_conv_2(x)

        # Step 6: reshape back to (B*F, H, W, C)
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        return x


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels, t_shape):
        super().__init__()
        self.in_channels = in_channels
        self.t_shape = t_shape
        self.norm = nn.BatchNorm3d(in_channels)
        self.q = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        h_ = x
        h_ = rearrange(h_, "(B F) C H W -> B C F H W", F=self.t_shape)
        print(h_.shape)
        b, c, f, h, w = h_.shape
        # x: shape (B*F,C,H,W)

        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # # compute attention
        # b, c, f, h, w = q.shape
        q = rearrange(q, "B C F H W -> (B C) F (H W)")
        # q = q.reshape(b, f,c*h*w)
        q = q.permute(0, 2, 1)  # bc,hw,f
        # k = k.reshape(b, f, c*h*w) # bc,f,hw
        k = rearrange(k, "B C F H W -> (B C) F (H W)")
        w_ = torch.bmm(q, k)  # bc,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(f) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # # attend to values
        # v = v.reshape(b,f,c*h*w) # b,f, chw
        v = rearrange(v, "B C F H W -> (B C) F (H W)")
        w_ = w_.permute(0, 2, 1)  # bc,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # bc,f,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_ = h_.reshape(b, c, f, h, w)
        # h_ = h_.permute(0,2,1,3,4) # b c f h w
        h_ = self.proj_out(h_)

        h_ = rearrange(h_, "B C F H W -> (B F) C H W")

        return x + h_