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

from model.ATFN import ATFN
from utils.CUDAinit import CUDA_Init

from configure.config import Model_ATFN_Config
from utils.metrics import Metrics
from utils.ModelUtils import count_parameters
from torch.utils.data import Dataset,DataLoader
from Data.Dataloader.AudioDataloader import AudioDataLoader

import torch
import torch.nn as nn
from torch import optim
from torch.utils.mobile_optimizer import optimize_for_mobile

class ATFN_Evaluator():
    # can only load ATFN
    def __init__(self,model_save_dir,dataloader):
        device_name,self.device = CUDA_Init()
        # model
        self.model= ATFN().to(self.device)
        self.model_config = Model_ATFN_Config()
        self.mode = self.model_config.mode
        # load dict
        assert os.path.exists(model_save_dir)
        try:
            self.model_dict = self.model.load_state_dict(torch.load(model_save_dir))
        except:
            raise Exception("Cannot load model from {}, please check the model structure".format(model_save_dir))
        
        print("This Model has {} parameters".format(count_parameters(self.model)))

        self.dataloader = dataloader
    
    def do_evaluate(self):
        # returns pred,true
        self.model.eval()
        y_pred,y_true = list(),list()
        
        with torch.no_grad():
            with tqdm(self.dataloader['test']) as td:
                  
                  for batch_data in td:
                    input = batch_data['feature'].to(self.device)
                    lbl = (batch_data['reg_lbl'] if self.mode=='reg' else batch_data['cls_lbl']).to(self.device)
                    # output
                    outputs = self.model(input)
                    y_pred.append(outputs.cpu())
                    y_true.append(lbl.cpu())
        
        preds, trues = torch.cat(y_pred), torch.cat(y_true)

        return (preds,trues)

    def do_evaluate_from_features(self,feature_in):
        # feature_in must in shape ( num_samples, feature_dim)
        # returns preds
        assert (len(feature_in.shape)==2) and \
            feature_in.shape[-1]==self.model_config.ModelParas["feature_dim"]
        # dataloader
        DL = self.make_dataloader(feature_in)
        preds = list()
        with torch.no_grad():
            with tqdm(DL) as td:
                  for batch_data in td:
                    input = batch_data.to(self.device)
                    # output
                    outputs = self.model(input)
                    preds.append(outputs.cpu())
        
        preds = torch.cat(preds)
        # flatten
        preds = preds.view(-1).cpu().detach().numpy()
               
        return (preds)
        
    def make_dataloader(self,feature_in):
        DS = FeatureDataset(feature_in)
        return DataLoader(DS,batch_size=128,shuffle=False)     
    
    
    
class FeatureDataset(Dataset):
    def __init__(self,feature):
        assert len(feature.shape)==2
        self.feature = feature
        
    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, index):
        return self.feature[index]
    

if __name__ == '__main__':
    metrics = Metrics('reg')
    model_save_dir = ('/home/tongyuang/code/VER_Health_Monitoring/results/best_models/2021-11-29-05-09-12-ATFN-best.pt')
    dataloader = AudioDataLoader()
    evaluator = ATFN_Evaluator(model_save_dir,dataloader)
    
    preds,truths = evaluator.do_evaluate()
    #print(preds,truths)
    metrics_dict = metrics.eval_regression(preds,truths)
    
    print(metrics_dict)