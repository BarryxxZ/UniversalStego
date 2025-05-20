# --------------------------------------------------------
# Universal Low Bit-Rate Speech Steganalysis
# Licensed under The MIT License
# Code written by Yiqin Qiu
# --------------------------------------------------------

import os
import torch
import random
import numpy as np
import torch.nn as nn


def set_seed(seed):
    """
    set seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class OrthoLoss(nn.Module):
    """
    Orthogonal loss function
    """
    def __init__(self):
        super(OrthoLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss
