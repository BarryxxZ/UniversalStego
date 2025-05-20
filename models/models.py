# --------------------------------------------------------
# Universal Low Bit-Rate Speech Steganalysis
# Licensed under The MIT License
# Code written by Yiqin Qiu
# --------------------------------------------------------

import os
import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from models.modules import *


class Backbone(nn.Module):
    """Separate Backbone of Matching Identification Network

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels

    """
    def __init__(self, in_channels=64, out_channels=256):
        super(Backbone, self).__init__()
        self.res_block1 = ResBlock(in_channels=in_channels, out_channels=128, re_sample=True)

        self.res_block2 = ResBlock(in_channels=128, out_channels=128)
        self.res_block3 = ResBlock(in_channels=128, out_channels=128)
        self.res_block4 = ResBlock(in_channels=128, out_channels=256, re_sample=True)

        self.res_block5 = ResBlock(in_channels=256, out_channels=256)
        self.res_block6 = ResBlock(in_channels=256, out_channels=out_channels)

        self.attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=4, dropout=0.2, add_bias_kv=True)

        self.ffn = IFFN(out_channels, 0.4)

        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)

    def forward(self, x):

        out = self.res_block1(x)
        out = self.res_block2(out)
        out = self.res_block3(out)

        out = self.res_block4(out)
        out = self.res_block5(out)
        out = self.res_block6(out)
        out1 = out.permute(0, 2, 1)

        out = self.ln1(out1)
        out = out.permute(1, 0, 2)
        out, _ = self.attn(out, out, out)

        out2 = out.permute(1, 0, 2) + out1

        out = self.ln2(out2)
        out = self.ffn(out)

        out = out + out2

        return out


class MatchingIdentiModel(nn.Module):
    """
    Matching Identification Network
    """

    def __init__(self):
        super(MatchingIdentiModel, self).__init__()
        self.qis_emb = nn.Embedding(512, 16)
        self.pos_emb = nn.Embedding(40, 16)
        self.int_emb = nn.Embedding(144, 32)
        self.frac_emb = nn.Embedding(6, 32)

        self.qis_backbone = Backbone(80, 256)
        self.pos_backbone = Backbone(160, 256)
        self.lag_backbone = Backbone(64, 256)

        self.domain_match = CrossDomainMatching(256)

        self.linear1 = nn.Linear(in_features=768, out_features=512)
        self.linear2 = nn.Linear(in_features=1024, out_features=2)
        self.linear_d = nn.Linear(in_features=256, out_features=2)
        self.gelu = nn.GELU()
        self.ffn1 = IFFN(256, 0.4)
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(768)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x1, x2):
        qis = x1[:, :, :5]
        pos = x1[:, :, 5:45]
        int_lag = x1[:, :, 45:49]
        frac_lag = x1[:, :, 49:]

        qis = self.qis_emb(qis)
        qis = qis.reshape(qis.shape[0], qis.shape[1], qis.shape[2] * qis.shape[3])

        pos = pos.reshape(pos.shape[0], pos.shape[1] * 4, pos.shape[2] // 4)
        pos = self.pos_emb(pos)
        pos = pos.reshape(pos.shape[0], pos.shape[1], pos.shape[2] * pos.shape[3])

        int_lag = int_lag.reshape(int_lag.shape[0], int_lag.shape[1] * int_lag.shape[2])
        int_lag = self.int_emb(int_lag)
        frac_lag = frac_lag.reshape(frac_lag.shape[0], frac_lag.shape[1] * frac_lag.shape[2])
        frac_lag = self.frac_emb(frac_lag)
        lag = torch.cat((int_lag, frac_lag), dim=2)

        qis = self.qis_backbone(qis.permute(0, 2, 1))
        pos = self.pos_backbone(pos.permute(0, 2, 1))
        lag = self.lag_backbone(lag.permute(0, 2, 1))

        out2_d, scores = self.domain_match(qis, pos, lag)

        out_d = self.ln2(out2_d)
        out_d = self.ffn1(out_d)

        out_d = out_d + out2_d

        out_d = self.avg_pool(out_d.permute(0, 2, 1))
        out = out_d.view(out_d.shape[0], out_d.shape[1] * out_d.shape[2])

        out_d = self.linear_d(out)
        return out_d, out


class DMSM(nn.Module):
    """
    Backbone of Content Alignment Network
    Use end-to-end training DMSM here
    """

    def __init__(self):
        super(DMSM, self).__init__()
        self.emb_k1 = nn.Embedding(512, 5)
        self.emb_k2 = nn.Embedding(512, 10)
        self.emb_k3 = nn.Embedding(512, 15)

        self.lstm = nn.LSTM(input_size=1590, hidden_size=300, num_layers=2, batch_first=True, dropout=0.3,
                            bidirectional=True)

        self.attn = nn.MultiheadAttention(600, 10)

        self.conv = nn.Conv1d(in_channels=600, out_channels=600, kernel_size=1)
        self.depth_conv = nn.Conv1d(in_channels=1200, out_channels=600, kernel_size=3, padding=1, groups=600)
        self.point_conv = nn.Conv1d(in_channels=600, out_channels=64, kernel_size=1)

        self.eca = EcaLayer(64)

        self.linear = nn.Linear(in_features=64, out_features=1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x1):
        k1 = self.emb_k1(x1)
        k1 = k1.view(k1.shape[0], k1.shape[1], k1.shape[2] * k1.shape[3])

        k2 = self.emb_k2(x1)
        k2 = k2.view(k2.shape[0], k2.shape[1], k2.shape[2] * k2.shape[3])

        k3 = self.emb_k3(x1)
        k3 = k3.view(k3.shape[0], k3.shape[1], k3.shape[2] * k3.shape[3])

        out = torch.cat((k1, k2, k3), dim=2)

        self.lstm.flatten_parameters()
        corr, _ = self.lstm(out)

        out = corr.permute(1, 0, 2)
        out, _ = self.attn(out, out, out)

        res = self.conv(corr.permute(0, 2, 1))
        out = torch.cat((out.permute(1, 2, 0), res), dim=1)
        out = self.point_conv(self.depth_conv(out))
        out = self.eca(out)
        out = self.avg_pool(out)
        out1 = out.view(out.shape[0], out.shape[1] * out.shape[2])

        return out1


class ContentAlignModel(nn.Module):
    """
    Content Alignment Network
    """

    def __init__(self):
        super(ContentAlignModel, self).__init__()
        self.backbone = DMSM()
        self.linear1 = nn.Linear(in_features=128, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x1, x2):
        out1 = self.backbone(x1)
        out2 = self.backbone(x2)
        diff = F.normalize(out1, dim=1) - F.normalize(out2, dim=1)
        out = self.linear2(diff)
        return out, diff


class CombineNet(nn.Module):
    """Separate Backbone of Matching Identification Network

    Args:
        path_identify: model weight path of MIN
        path_align: model weight path of CAN

    """
    def __init__(self, path_identify, path_align):
        super(CombineNet, self).__init__()
        align_net = ContentAlignModel()
        align_net.load_state_dict(torch.load(path_align))
        self.align_net = align_net

        identi_net = MatchingIdentiModel()
        identi_net.load_state_dict(torch.load(path_identify))
        self.identi_net = identi_net

        for p in self.parameters():
            p.requires_grad = False

        self.linear1 = nn.Linear(in_features=320, out_features=128)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x1, x2):
        _, out1 = self.identi_net(x1, x2)
        _, diff = self.align_net(x1, x2)

        out2 = torch.cat((out1, diff), dim=1)

        out = self.gelu(self.linear1(out2))
        out_total = self.linear2(out)

        return out_total, out2
