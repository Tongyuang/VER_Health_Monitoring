#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ACN.py
@Last Modified    :   2021/12/01 11:56:48
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

from torch.nn.modules import conv
sys.path.append('../../')

from configure.config import Model_ACN_Config

class SubConvNet(nn.Module):
    '''
    sub 1-D conv Net
    '''
    def __init__(self,in_channels,out_channels,conv_kernel_size,pooling_kernel_size,dropout,activation):
        # input shape: (batch_size,C_in,feature_dim)
        # output shape: (batch_size,C_out,feature_dim/2)
        # stride = 1
        # padding to keep the feature_dim 
        super(SubConvNet,self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        # must be odd number
        assert self.conv_kernel_size%2 == 1
        
        self.stride = 1
        self.padding = int((self.conv_kernel_size-1)/2)
        self.padding_mode = 'zeros'
        
        self.pooling_kernel_size = pooling_kernel_size
        
        self.conv1 = nn.Conv1d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = self.conv_kernel_size,
            padding = self.padding,
            padding_mode= self.padding_mode,
            stride = self.stride)
        
        self.conv2 = nn.Conv1d(
            in_channels = self.out_channels,
            out_channels = self.out_channels,
            kernel_size = self.conv_kernel_size,
            padding = self.padding,
            padding_mode= self.padding_mode,
            stride = self.stride)

        self.pooling = nn.MaxPool1d(
            kernel_size=self.pooling_kernel_size,
            stride = self.pooling_kernel_size
        )
        
        self.norm = nn.BatchNorm1d(self.in_channels)
        self.drop = nn.Dropout(p=dropout)
        
        self.activation_mode = activation
        assert self.activation_mode in ['relu','leaky_relu','tanh']
        
        if self.activation_mode == 'relu':
            self.activation = F.relu
        elif self.activation_mode == 'leaky_relu':
            self.activation = F.leaky_relu
        elif self.activation_mode == 'tanh':
            self.activation = torch.tanh
                    
    def forward(self,x):
        '''
        Args:
            x: tensor of shape (batch_size,in_channels,in_size) # in_size=feature_dim
        '''
        
        normed = self.norm(x)
        dropped = self.drop(normed)
        
        y = self.conv1(dropped)
        y = self.conv2(y)
        y = self.activation(self.pooling(y))
        
        return y

class ACN(nn.Module):
    '''
    ACN: Audio Convolutional Net
    '''
    def __init__(self):
        super(ACN,self).__init__()
        
        self.ModelConfig = Model_ACN_Config()
        self.feature_dim = self.ModelConfig.ModelParas['feature_dim']
        self.mode = self.ModelConfig.mode
        self.channels = self.ModelConfig.ModelParas['channels']
        self.conv_layers = nn.Sequential()
        self.conv_layer_init()
            
        self.output_dropout = nn.Dropout(p=self.ModelConfig.ModelParas['dropout'])
        self.output_dim = self.ModelConfig.ModelParas['num_classes'] if self.mode=='cls' else 1

        self.output_hidden_dim = self.ModelConfig.ModelParas['output_hidden_dim']
        self.output_layer = nn.Linear(self.output_hidden_dim,self.output_dim)
            
        self.output_activation_mode = self.ModelConfig.ModelParas['output_activation']
        if self.output_activation_mode == 'relu':
            self.output_activation = F.relu
        elif self.output_activation_mode == 'leaky_relu':
            self.output_activation = F.leaky_relu
        elif self.output_activation_mode == 'tanh':
            self.output_activation = torch.tanh
            
    def conv_layer_init(self):
        assert len(self.channels) > 0
        self.conv_layers.add_module('conv_0',SubConvNet(
            in_channels=1,
            out_channels=self.channels[0],
            conv_kernel_size=self.ModelConfig.ModelParas['kernel_size'],
            pooling_kernel_size=self.ModelConfig.ModelParas['pooling_kernel_size'],
            dropout=self.ModelConfig.ModelParas['dropout'],
            activation= self.ModelConfig.ModelParas['activation']        
        ))
        for idx in range(len(self.channels)-1):
            self.conv_layers.add_module('conv_{0}'.format(idx+1),SubConvNet(
            in_channels=self.channels[idx],
            out_channels=self.channels[idx+1],
            conv_kernel_size=self.ModelConfig.ModelParas['kernel_size'],
            pooling_kernel_size=self.ModelConfig.ModelParas['pooling_kernel_size'],
            dropout=self.ModelConfig.ModelParas['dropout'],
            activation= self.ModelConfig.ModelParas['activation']
                
        ))
        
    def forward(self,x):
        '''
        Args:
            x: tensor of shape (batch_size, 1, in_size) # in_size=feature_dim
        '''
        x_h = self.conv_layers(x)
        # (batch_size,out_channels,out_size)
        x_h = torch.flatten(x_h,1,-1)
        # (batch_size,out_channels*out_size)
            
        y = self.output_dropout(x_h)

        y = self.output_activation(self.output_layer(y))    
            
        return y