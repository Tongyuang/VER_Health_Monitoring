#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Last Modified    :   2021/11/30 10:22:14
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

import os
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../')

import time
from utils.CUDAinit import CUDA_Init
from model.ModelSelector import ModelSelector
from utils.metrics import Metrics
from utils.ModelUtils import count_parameters
from torch.utils.data import Dataset,DataLoader
from Data.Dataloader.AudioDataloader import AudioDataLoader

import configure.config as cfg
import torch
import torch.nn as nn
from torch import optim



class ATFN_Evaluator():
    
    def __init__(self,model_save_dir,model_name):
        device_name,self.device = CUDA_Init()
        # model
        self.model = ModelSelector(model_name)
        self.model.to(self.device)
        if self.model_name == 'ALSTM':
            self.model_config = cfg.Model_ALSTM_Config()
        elif self.model_name == 'ATFN':
            self.model_config = cfg.Model_ATFN_Config()
        elif self.model_name == 'ACN':
            self.model_config = cfg.Model_ACN_Config()

        self.mode = self.model_config.mode
        # load dict
        assert os.path.exists(model_save_dir)
        try:
            self.model_dict = self.model.load_state_dict(torch.load(model_save_dir))
        except:
            raise Exception("Cannot load model from {}, please check the model structure".format(model_save_dir))
        
        print("This Model has {} parameters".format(count_parameters(self.model)))
    
    def forward(self,input):
        '''
        input: feature of shape(1, feature_dim)
        output: (1,) if reg mode, (num_classes,) if cls mode
                inference time (in sec)
        '''
        self.model.eval()
        input = input.to(self.device)
        tic = time.time()
        output = self.model(input)
        output = output.cpu()
        toc = time.time()
        return output,(toc-tic)
    
    
    

if __name__ == '__main__':
    metrics = Metrics('reg')
    model_save_dir = ('/home/tongyuang/code/VER_Health_Monitoring/results/best_models/2021-11-29-05-09-12-ATFN-best.pt')
    dataloader = AudioDataLoader()
    evaluator = ATFN_Evaluator(model_save_dir,dataloader)
    
    preds,truths = evaluator.do_evaluate()
    #print(preds,truths)
    metrics_dict = metrics.eval_regression(preds,truths)
    
    print(metrics_dict)