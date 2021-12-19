#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   GenLiteDataset.py
@Last Modified    :   2021/12/16 18:19:41
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
@Usage   :   gen a lite dataset for test
'''

# here put the import lib

import sys

import pickle
import numpy  as np
import os
sys.path.append('../../')

from configure.config import DataPreConfig


class LiteDatasetGenerator():
    
    def __init__(self,volumn):
        '''
        args:
            volumns: lite dataset's size(dict)
        '''
        super(LiteDatasetGenerator,self).__init__
        self.cfg = DataPreConfig()
        self.dataset_names = self.cfg.dataset_names
        self.split_ratio = self.cfg.split_ratio
        self.raw_data_dir = self.cfg.feature_store_dir
        self.suffix = self.cfg.suffix
        self.volumn = volumn
        self.modes = ['train','valid','test']       
        self.liteDataset = {name:{} for name in self.dataset_names}
        self.DatasetKeys = ['feature', 'num_samples', 'reg_lbls', 'cls_lbls']
        self.savedir = {
            name:os.path.join(self.cfg.DATA_dir,name,'feature_all_lite'+self.suffix) for name in self.dataset_names
        }
        self.GenLiteDataset()
    def GenLiteDataset(self):
        '''
        Gen a lite dataset
        '''
        for name in self.dataset_names:
            Feature_Dir = self.raw_data_dir[name]
            with open(Feature_Dir,'rb') as rf:
                data = pickle.load(rf)
                
            # cal_num
            for mode in self.modes:
                mode_length = int(self.volumn[name]*self.split_ratio[mode])
                # select from raw data
                self.liteDataset[name][mode] = {}
                for datasetkey in self.DatasetKeys:
                    
                    if datasetkey == 'num_samples':
                        self.liteDataset[name][mode][datasetkey] = mode_length
                    else:
                        self.liteDataset[name][mode][datasetkey] = data[mode][datasetkey][0:mode_length,:].copy()

                print("Dataset name: {}, mode: {}, shape:{}".format(name,mode, self.liteDataset[name][mode]['feature'].shape))
    def Save(self):
        for name in self.dataset_names:
            output_file = self.savedir[name] # which file to save？
            if os.path.exists(output_file):
                os.remove(output_file)            
            
            try:
                with open(output_file,'wb') as wf:
                    pickle.dump(self.liteDataset[name],wf)
                print('Lite features are saved at {}'.format(output_file))
            
            except:
                print('Cannot Save Feature at {}, cannot open target file.'.format(output_file))

if __name__=="__main__":
    
    volumn = {
        "IEMOCAP":100,
        "CH_SIMS":100,
    }
    
    generator = LiteDatasetGenerator(volumn)
    generator.Save()
                
                
            