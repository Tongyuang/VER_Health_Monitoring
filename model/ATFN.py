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

    def __init__(self, in_size, out_size, dropout,activation):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        
        self.norm = nn.BatchNorm1d(out_size)
        self.drop = nn.Dropout(p=dropout)
        
    
        self.activation_mode = activation
        
        assert self.activation_mode in  ['relu','leaky_relu','tanh']
        
        if self.activation_mode == 'relu':
            self.activation = F.relu
        elif self.activation_mode == 'leaky_relu':
            self.activation = F.leaky_relu
        elif self.activation_mode == 'tanh':
            self.activation = torch.tanh
            
        self.linear = nn.Linear(in_size,out_size)
        
    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size) # in_size=feature_dim
        '''
        
        y = self.linear(x)
        y = self.norm(y)
        y = self.activation(y)
        y = self.drop(y)
        return y

class ATFN(nn.Module):
    def __init__(self):
        super(ATFN,self).__init__()
        self.ModelConfig = Model_ATFN_Config()
        self.mode = self.ModelConfig.mode
        assert self.mode in ['reg','cls']
        
        self.feature_dim = self.ModelConfig.ModelParas['feature_dim']
        self.hidden_dims = self.ModelConfig.ModelParas['hidden_dims']
        self.dropout = self.ModelConfig.ModelParas['dropout']
        self.post_dropout = self.ModelConfig.ModelParas['post_dropout']
        self.output_dim = self.ModelConfig.ModelParas['num_classes'] if self.mode=='cls' else 1
        
        self.activation_mode = self.ModelConfig.ModelParas['activation']
        
        assert self.activation_mode in  ['relu','leaky_relu','tanh']
        
        if self.mode=='reg':
            self.output_activation_mode = self.ModelConfig.ModelParas['output_activation']
        else:#classification
            self.output_activation_mode = self.ModelConfig.ModelParas['output_activation_for_classification']
        
        assert self.output_activation_mode in  ['relu','leaky_relu','tanh']
            
        self.layers = nn.Sequential()
        
        self.__init__layers()
    
    def __init__layers(self):
        '''
        init self layers
        '''
        cur_dim = self.feature_dim
        for i,dim in enumerate(self.hidden_dims):
            self.layers.add_module('linear_{}'.format(i),SubNet(
                in_size=cur_dim,
                out_size=dim,
                dropout=self.dropout,
                activation=self.activation_mode
            ))
            cur_dim = dim
        
        self.layers.add_module('linear_{}'.format(len(self.hidden_dims)),SubNet(
            in_size = cur_dim,
            out_size = self.output_dim,
            dropout = self.post_dropout,
            activation = self.output_activation_mode
        ))
        
    def forward(self,x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size) # in_size=feature_dim
        '''
        
        return self.layers(x)
        