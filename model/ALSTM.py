#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ALSTM.py
@Last Modified    :   2021/12/01 18:08:10
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib
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
        y,(h_n,c_n) = self.rnn(x)
        return y,(h_n,c_n)
        
        
class ALSTM(nn.Module):
    '''
    ALSTM: Audio LSTM
    '''
    def __init__(self):

        # output shape:(batch_size,)
        super(ALSTM,self).__init__()
        self.ModelConfig = Model_ALSTM_Config()
        
        self.mode = self.ModelConfig.mode
        assert self.mode in ['reg','cls']
        
        