"""
Ref paper: Tensor Fusion Network for Multimodal Sentiment Analysis
Ref url: https://github.com/Justin1904/TensorFusionNetworks
"""

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ATFN.py
@Last Modified    :   2021/11/28 19:43:22
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

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
        
        self.ModelConfig = Model_ATFN_Config()
        
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
    
        self.activation_mode = self.ModelConfig.SubnetParas['activation']
        
        assert self.activation_mode in  ['relu','leaky_relu','tanh']
        
        if self.activation_mode == 'relu':
            self.activation = F.relu
        elif self.activation_mode == 'leaky_relu':
            self.activation = F.leaky_relu
        elif self.activation_mode == 'tanh':
            self.activation = torch.tanh
            
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
        y_1 = self.activation(self.linear_1(dropped))
        y_2 = self.activation(self.linear_2(y_1))
        y_3 = self.activation(self.linear_3(y_2))

        return y_3

class ATFN(nn.Module):
    def __init__(self):
        super(ATFN,self).__init__()
        self.ModelConfig = Model_ATFN_Config()
        self.mode = self.ModelConfig.mode
        assert self.mode in ['reg','cls']
        
        self.feature_dim = self.ModelConfig.ModelParas['feature_dim']
        self.hidden_dim = self.ModelConfig.ModelParas['hidden_dim']
        self.dropout = self.ModelConfig.ModelParas['dropout']
        
        self.post_hidden_dim = self.ModelConfig.ModelParas['post_hidden_dim']
        self.post_dropout = self.ModelConfig.ModelParas['post_drouput']
        
        self.output_dim = self.ModelConfig.ModelParas['num_classes'] if self.mode=='cls' else 1
        
        
        self.activation_mode = self.ModelConfig.ModelParas['activation']
        
        assert self.activation_mode in  ['relu','leaky_relu','tanh']
        
        if self.activation_mode == 'relu':
            self.activation = F.relu
        elif self.activation_mode == 'leaky_relu':
            self.activation = F.leaky_relu
        elif self.activation_mode == 'tanh':
            self.activation = F.tanh
        
        self.output_activation_mode = self.ModelConfig.ModelParas['output_activation']
        
        assert self.output_activation_mode in  ['relu','leaky_relu','tanh']
        
        if self.output_activation_mode == 'relu':
            self.output_activation = F.relu
        elif self.output_activation_mode == 'leaky_relu':
            self.output_activation = F.leaky_relu
        elif self.output_activation_mode == 'tanh':
            self.output_activation = torch.tanh
            
        
        self.subnet = SubNet(self.feature_dim,self.hidden_dim, self.dropout)
        
        self.post_dropout = nn.Dropout(self.post_dropout)
        self.post_linear1 = nn.Linear(self.hidden_dim,self.post_hidden_dim)
        self.post_linear2 = nn.Linear(self.post_hidden_dim,self.post_hidden_dim)
        self.output_layer = nn.Linear(self.post_hidden_dim,self.output_dim)
    
    def forward(self,x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size) # in_size=feature_dim
        '''
        if len(x.shape)>2:
            x = x.squeeze(1)
        
        x_h = self.subnet(x)
        
        x_a = self.post_dropout(x_h)
        x_a = self.activation(self.post_linear1(x_a),inplace=True)
        x_a = self.activation(self.post_linear2(x_a),inplace=True)
        x_o = self.output_activation(self.output_layer(x_a))
        
        return x_o
        