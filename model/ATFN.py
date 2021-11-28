"""
Ref paper: Tensor Fusion Network for Multimodal Sentiment Analysis
Ref url: https://github.com/Justin1904/TensorFusionNetworks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../../')

from configure.config import Model_ATFN_Config

__all__ = ['SubNet', 'TextSubNet']

class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size) # in_size=feature_dim
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3

class ATFN(nn.Module):
    def __init__(self):
        super(ATFN,self).__init__()
        self.ModelConfig = Model_ATFN_Config()
        self.mode = self.ModelConfig.mode
        assert self.mode in ['reg','cls']
        
        self.feature_dim = self.ModelConfig.datasetParas['feature_dim']
        self.dropout = self.ModelConfig.datasetParas['dropout']
        self.post_dropout = self.ModelConfig.datasetParas['post_drouput']
        
        self.output_dim = self.ModelConfig.datasetParas['num_classes'] if self.mode=='cls' else 1
        