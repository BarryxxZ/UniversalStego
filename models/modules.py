# --------------------------------------------------------
# Universal Low Bit-Rate Speech Steganalysis
# Licensed under The MIT License
# Code written by Yiqin Qiu
# --------------------------------------------------------

import math
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv1d(nn.Module):
    """Separate convolution layer

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size
        padding (int): Padding size
        bias (bool): bias or not

    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True):
        super(SeparableConv1d, self).__init__()
        self.depth_conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.point_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depth_conv(x)

        out = self.point_conv(out)

        return out


class EcaLayer(nn.Module):
    """Constructs an ECA module used in DMSM backbone.

    Args:
        channels (int): Number of channels of the input feature map
        gamma: parameter of mapping function
        b: parameter of mapping function
    """
    def __init__(self, channels, gamma=2, b=1):
        super(EcaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(channels, channels, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        y = self.conv(y).transpose(-1, -2)

        y = self.sigmoid(y)

        return x * y.transpose(-1, -2)


class PAEM(nn.Module):
    """Pivotal-Feature Adaptive Enhancement Module

    Args:
        in_channels (int): Number of input channels

    """
    def __init__(self, in_channels):
        super(PAEM, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 2, bias=False)
        self.fc2 = nn.Linear(in_channels // 2, in_channels, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1, bias=False)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        spatial_scale = self.sigmoid1(self.conv(x))
        out = x * spatial_scale

        channel_squeeze = self.pool(out)
        channel_squeeze = self.sigmoid2(self.fc2(self.relu(self.fc1(channel_squeeze.permute(0, 2, 1)))))
        channel_squeeze = channel_squeeze.permute(0, 2, 1)

        out = out * channel_squeeze

        return out


class ResBlock(nn.Module):
    """Constructs Residual Block

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        re_sample (bool): whether to resample residual features

    """
    def __init__(self, in_channels, out_channels, re_sample=False):
        super(ResBlock, self).__init__()
        self.re_sample = re_sample
        self.leru = nn.GELU()
        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(out_channels)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.pae = PAEM(out_channels)
        self.conv1 = SeparableConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = SeparableConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        if re_sample:
            self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.bn1(x)
        out = self.leru(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.leru(out)
        out = self.conv2(out)

        if self.re_sample:
            short_cut = self.conv3(x)
        else:
            short_cut = x

        out = self.pae(out)
        out = out + short_cut

        return out


class CrossDomainMatching(nn.Module):
    """Cross-Domain Matching Module

    Args:
        dim (int): Number of input dimension
        heads (int): number of heads, extended as the mechanism of multi-head attention
        dim_head (int): number of dimension inside each head
        dropout (float): dropout probability

    Returns:
        self.to_out(out) + fea: output feature map
        match_score.mean(dim=1): normalized Gram matrix, i.e., matching scores

    """
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.2):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.drop = dropout
        self.scale = math.sqrt(1.0 / float(dim_head))

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ln = nn.LayerNorm(256)
        self.norm = nn.Softmax(dim=-1)

        self.linear_theta = nn.Linear(dim, inner_dim, bias=True)
        self.linear_phi = nn.Linear(dim, inner_dim, bias=True)
        self.linear_mu = nn.Linear(dim, inner_dim, bias=True)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

        # Register used to allocate scores in cover samples, inspired by Vision Transformer need Registers
        self.register_phi = torch.nn.Parameter(torch.empty((1, 1, inner_dim)))
        self.register_mu = torch.nn.Parameter(torch.empty((1, 1, inner_dim)))
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.register_phi, std=0.02)
        nn.init.trunc_normal_(self.register_mu, std=0.02)

    def forward(self, x1, x2, x3):
        x1 = self.pool(x1.permute(0, 2, 1))
        x2 = self.pool(x2.permute(0, 2, 1))
        x3 = self.pool(x3.permute(0, 2, 1))

        x = torch.cat((x1, x2, x3), dim=2)
        fea = x.permute(0, 2, 1)

        x = self.ln(fea)
        x_theta = self.linear_theta(x)
        x_phi = self.linear_phi(x)
        x_mu = self.linear_mu(x)

        x_phi = torch.cat([x_phi, self.register_phi.repeat(x_phi.size(0), 1, 1)], dim=1)
        x_mu = torch.cat([x_mu, self.register_mu.repeat(x_mu.size(0), 1, 1)], dim=1)

        x_theta = einops.rearrange(x_theta, 'b n (h d) -> b h n d', h=self.heads)
        x_phi = einops.rearrange(x_phi, 'b n (h d) -> b h n d', h=self.heads)
        x_mu = einops.rearrange(x_mu, 'b n (h d) -> b h n d', h=self.heads)

        x_theta = x_theta * self.scale
        gram_fea = torch.matmul(x_theta, x_phi.transpose(-1, -2))
        match_score = self.norm(gram_fea)
        match_score = F.dropout(match_score, p=self.drop, training=self.training)

        out = torch.matmul(match_score, x_mu)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out) + fea, match_score.mean(dim=1)


class IFFN(nn.Module):
    """Inverted FFN module

    Args:
        in_channels (int): Number of input channels
        drop(float): dropout probability

    """
    def __init__(self, in_channels, drop=0.0):
        super(IFFN, self).__init__()
        self.linear1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1)
        self.linear2 = nn.Conv1d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1)
        self.conv = nn.Conv1d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=3, padding=1, groups=in_channels * 2)

        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()

        self.drop_out = nn.Dropout(drop)

    def forward(self, x):
        out = self.gelu1(self.linear1(x.permute(0, 2, 1)))
        out = self.drop_out(out)

        out = self.gelu2(self.conv(out))
        out = self.linear2(out)
        out = self.drop_out(out)

        return out.permute(0, 2, 1)
