#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   AResN.py
@Time    :   2021/12/15 21:19:47
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append('../../')

from configure.config import Model_ACN_Config

class subResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_kernel_size,
                 pooling_kernel_size,
                 dropout,
                 activation):
    # input shape: (batch_size,C_in,feature_dim)
    # output shape: (batch_size,C_out,feature_dim/2)
    # stride = 1
    # padding to keep the feature_dim 
        super(subResBlock,self).__init__()
    