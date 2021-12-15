#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ALSTM.py
@Last Modified    :   2021/12/01 18:08:10
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../../')

from configure.config import Model_ALSTM_Config

class subLSTM(nn.Module):
    '''
    subNet: LSTM
    '''
    def __init__(self):
        super(subLSTM,self).__init__()
        self.ModelConfig = Model_ALSTM_Config()
        self.feature_dim = self.ModelConfig.ModelParas['feature_dim']
        self.num_layers = self.ModelConfig.ModelParas['num_layers']
        self.hidden_size = self.ModelConfig.ModelParas['hidden_size']
        
        self.rnn = nn.LSTM(self.feature_dim,self.hidden_size,self.num_layers)
    
    def forward(self,x):
        # input shape:(seq_len,batch_size,emb_dim)
        # LSTM output: y: (seq_len,batch_size,hidden_size)
        #              h_n: (num_layers,batch_size,hidden_size)
        #              c_n: (num_layers,batch_size,hidden_size)
        y,(h_n,c_n) = self.rnn(x.float())
        return y,(h_n,c_n)
        
        
class ALSTM(nn.Module):
    '''
    ALSTM: Audio LSTM
    '''
    def __init__(self):

        # output shape:(batch_size,)
        super(ALSTM,self).__init__()
        self.ModelConfig = Model_ALSTM_Config()
        self.subLSTM = subLSTM()
        self.hidden_size = self.ModelConfig.ModelParas['hidden_size']
        self.mode = self.ModelConfig.mode
        assert self.mode in ['reg','cls']
        
        self.feature_dim = self.ModelConfig.ModelParas['feature_dim']
        self.sequence_length = self.ModelConfig.ModelParas['sequence_length']
        self.drouput = self.ModelConfig.ModelParas['dropout']
        self.norm_mode = self.ModelConfig.ModelParas['norm_mode']
        assert self.norm_mode in ['last','mean']
        
        self.linear_hidden_dim = self.ModelConfig.ModelParas['linear_hidden_dim']
        
        self.output_dim = self.ModelConfig.ModelParas['num_classes'] if self.mode=='cls' else 1
        self.activation_mode = self.ModelConfig.ModelParas['output_activation']
        assert self.activation_mode in ['relu','leaky_relu','tanh']
        if self.activation_mode == 'relu':
            self.activation = F.relu
        elif self.activation_mode == 'leaky_relu':
            self.activation = F.leaky_relu
        elif self.activation_mode == 'tanh':
            self.activation = torch.tanh
        
        
        self.output_Layer = nn.Sequential(
            nn.Linear(self.hidden_size,self.linear_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.linear_hidden_dim,self.output_dim),
            nn.BatchNorm1d(self.output_dim),
        )
    
    def forward(self,x):
        '''
        input:
        x : tensor of shape (seq_len,batch_size,emb_dim)
        output:
        y : tensor of shape (batch_size,1)
        '''
        
        y_h,(h_n,c_n) = self.subLSTM(x) # y_h: (seq_len,batch_size,hidden_size)
        
        if self.norm_mode=='last':
            y_h = y_h[-1] # (batch_size,hidden_size)
        else: # mean
            y_h = torch.mean(y_h,dim=0)
        
        y = self.activation(self.output_Layer(y_h))
        
        return y