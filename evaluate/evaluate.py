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
import librosa
import sys
sys.path.append('../')

import time
from utils.CUDAinit import CUDA_Init
from model.ModelSelector import ModelSelector
from utils.metrics import Metrics
from utils.ModelUtils import count_parameters
from Data.Dataloader.AudioDataloader import AudioDataLoader
from utils import recorder
import configure.config as cfg
import torch


from Data.DataPre import DataPreProcessor


class Evaluator():
    
    def __init__(self,model_save_dir,model_name):
        device_name,self.device = CUDA_Init()
        # model
        self.model = ModelSelector(model_name)
        self.model_name = model_name
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
            self.model.load_state_dict(torch.load(model_save_dir),strict=True)
        except:
            raise Exception("Cannot load model from {}, please check the model structure".format(model_save_dir))
        
        print("This Model has {} parameters".format(count_parameters(self.model)))
    def forward(self,input):
        '''
        input: feature of shape(batch_size, feature_dim)
        output: (batch_size,1) if reg mode, (batch_size,num_classes) if cls mode
                inference time (in sec)
        '''
        self.model.eval()
        input = torch.tensor(input)
        input = input.to(self.device,dtype=torch.float32)
        input = input.view((input.shape[0],1,input.shape[-1]))
        
        tic = time.time()
        output = self.model(input)
        output = output.cpu()
        toc = time.time()
        return output,(toc-tic)
    
def preprocessor(wav_dir_list,feature_dim=562):
    '''
    args: wav_dir_list: list of absolute wav dir
    '''
    output = np.zeros((len(wav_dir_list),feature_dim))
    dpp = DataPreProcessor()
    for i,wav_dir in enumerate(wav_dir_list):
        assert os.path.exists(wav_dir)
        feature_out = np.mean(dpp.getSingleAudioEmbd(wav_dir),axis=0)
        output[i] = feature_out
    return output


            
    
    
    

if __name__ == '__main__':
    wav_list = ["../dataset/IEMOCAP/Ses01F_impro01_F000.wav"]
    input = preprocessor(wav_list)
    print(input.shape)
    
    model_save_dir = "/Users/tongyuang/study/graduate_yr1/course/SPDM_AIoT/bighw/source/VER/results/best_models/2021-12-19-06-31-35-ACN-cls-best.pt"
    model_name = "ACN"
    
    evaluator = Evaluator(model_save_dir=model_save_dir,
                          model_name = model_name)
    
    output,time_interval = evaluator.forward(input)
    
    print(output,time_interval)