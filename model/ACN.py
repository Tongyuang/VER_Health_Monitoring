#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ACN.py
@Last Modified    :   2021/12/01 11:56:48
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

class SubConvNet(nn.Module):
    '''
    sub 1-D conv Net
    '''
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
        self.dropout = dropout
        
        self.conv = nn.Conv1d( # input: (feature_dim,in_channels)
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = self.conv_kernel_size,
            padding = self.padding,
            padding_mode= self.padding_mode,
            stride = self.stride) # (feature_dim,out_channels)
        
        if self.pooling_kernel_size>0:
            self.pooling = nn.MaxPool1d(
                kernel_size=self.pooling_kernel_size,
                stride=self.pooling_kernel_size,
            )
        self.norm = nn.BatchNorm1d(self.out_channels)
        
        if self.dropout>0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
                
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
        
        y = self.conv(x)
        if self.pooling_kernel_size>0:
            y = self.pooling(y)
        y = self.norm(y)
        y = self.activation(y)
        if self.dropout>0:
            y = self.dropout_layer(y)
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
        # parameters
        self.channels = self.ModelConfig.ModelParas['channels']
        self.kernels = self.ModelConfig.ModelParas['kernels']
        self.pooling_kernels = self.ModelConfig.ModelParas['pooling_kernels']
        self.dropout_fracs = self.ModelConfig.ModelParas['dropout']
        self.conv_activation = self.ModelConfig.ModelParas['activation']
        self.conv_layers = nn.Sequential()
        self.conv_layer_init()
        
        # output    
        self.output_dropout = nn.Dropout(p=self.ModelConfig.ModelParas['dropout'])
        self.output_dim = self.ModelConfig.ModelParas['num_classes'] if self.mode=='cls' else 1
        self.output_hidden_dim = self.ModelConfig.ModelParas['output_hidden_dim']
        self.output_layer = nn.Linear(self.output_hidden_dim,self.output_dim)
        if self.mode=='reg':
            self.output_activation_mode = self.ModelConfig.ModelParas['output_activation']
        else:#classification
            self.output_activation_mode = self.ModelConfig.ModelParas['output_activation_for_classification']
            
        if self.output_activation_mode == 'relu':
            self.output_activation = F.relu
        elif self.output_activation_mode == 'leaky_relu':
            self.output_activation = F.leaky_relu
        elif self.output_activation_mode == 'tanh':
            self.output_activation = torch.tanh
            
    def conv_layer_init(self):
        assert len(self.channels) > 0
        assert len(self.channels)==len(self.kernels)
        assert len(self.channels)==len(self.pooling_kernels)
        assert len(self.channels)==len(self.dropout_fracs)
        
        cur_channel = 1
        for idx in range(len(self.channels)):
            self.conv_layers.add_module('conv_{}'.format(idx),SubConvNet(
                in_channels=cur_channel,
                out_channels=self.channels[idx],
                conv_kernel_size=self.kernels[idx],
                pooling_kernel_size=self.pooling_kernels[idx],
                dropout=self.dropout_fracs[idx],
                activation=self.conv_activation
            ))
            cur_channel = self.channels[idx]
            
        
    def forward(self,x):
        '''
        Args:
            x: tensor of shape (batch_size, 1, in_size) # in_size=feature_dim
        '''
        x_h = self.conv_layers(x)
        # (batch_size,out_channels,out_size)
        x_h = torch.flatten(x_h,1,-1)
        # (batch_size,out_channels*out_size)
        y = self.output_activation(self.output_layer(x_h))
        y = self.output_layer(y) 
            
        return y