#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   AudioDataloader.py
@Last Modified    :   2021/11/26 14:58:39
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib


import os
import pickle
import numpy as np
import torch

import logging

from torch.utils.data import Dataset,DataLoader, dataloader

import sys
sys.path.append('../../')

from configure.config import DataPreConfig,DataLoaderConfig

class AudioDataset(Dataset):
    def __init__(self,mode):
        # config
        self.config = DataPreConfig()
        assert mode in ['train','valid','test']
        self.mode = mode
        self.DatasetMap = dict()
        
        for dsname in self.config.dataset_names:
            self.LoadDataset(dsname)
        self.Concatenate()
        
    def LoadDataset(self,name):
        # Load The Datasets By Mode
        feature_dir = self.config.feature_store_dir[name]
        with open(feature_dir,'rb') as rf:
            full_feature = pickle.load(rf)
        
        self.DatasetMap[name] = full_feature[self.mode]
    
    def Normalize(self):
        # (num_samples,length,feature_dim) -> ( num_samples, feature_dim)
        for dsname in self.DatasetMap.keys():
            feature = self.DatasetMap[dsname]['feature']
            feature = np.transpose(feature,(1,0,2))
            feature = np.mean(feature,axis=0,keepdims=False)
            self.DatasetMap[dsname]['feature'] = feature
    
    def Concatenate(self):
        # Concatenate
        self.FullDataset = {}
        self.Normalize()
        for dsname in self.DatasetMap.keys():
            keys = self.DatasetMap[dsname].keys()
            for key in keys:
                if not key in self.FullDataset:
                    self.FullDataset[key] = self.DatasetMap[dsname][key]
                else:
                    val = self.DatasetMap[dsname][key]
                    if type(val)==int:
                        self.FullDataset[key] += val
                    elif type(val)==np.ndarray:
                        # (num_samples, feature_dim)
                        assert self.FullDataset[key].shape[-1]==val.shape[-1]
                        tmp = np.concatenate((self.FullDataset[key],val),axis=0)
                        self.FullDataset[key] = tmp.copy()
    

    def __len__(self):
        return self.FullDataset['num_samples']
    
    def __getitem__(self, index):
        
        feature = self.FullDataset['feature'][index]
        reg_lbl = self.FullDataset['reg_lbls'][index]
        cls_lbl = self.FullDataset['cls_lbls'][index]
        sample = {
            'feature': torch.Tensor(feature),
            'reg_lbl': torch.Tensor(reg_lbl),
            'cls_lbl':torch.Tensor(cls_lbl)
        }                 
        return sample

    def get_feature_dim(self):
        return self.FullDataset['feature'].shape[-1]

def AudioDataLoader():
    loadercfg = DataLoaderConfig()
    batch_size = loadercfg.BatchSize
    num_worker = loadercfg.num_worker
    
    datasets = {
        'train':AudioDataset('train'),
        'valid':AudioDataset('valid'),
        'test':AudioDataset('test')
    }
    
    FullDataLoader ={
        key: DataLoader(datasets[key],batch_size=batch_size,num_workers=num_worker,shuffle=True)
        for key in datasets.keys()
    }
    
    return FullDataLoader
     
if __name__ == '__main__':
    
    pass
      
    
        
        
        
        
        
