#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   GenAngerDataset.py
@Last Modified    :   2021/12/19 16:44:36
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib


import sys

import pickle
import numpy  as np
import os
sys.path.append('../../')

from configure.config import DataPreConfig


class AngerDatasetGenerator():
    def __init__(self,anger_frac):
        '''
        args:
            anger_frac: the frac of anger samples
        '''
        super(AngerDatasetGenerator,self).__init__
        self.cfg = DataPreConfig()
        self.dataset_names = self.cfg.dataset_names
        self.modes = ['train','valid','test']  
        self.raw_data_dir = self.cfg.feature_store_dir
        self.angerDataset = {name:{} for name in self.dataset_names}
        self.DatasetKeys = ['feature', 'num_samples', 'reg_lbls', 'cls_lbls']
        self.suffix = self.cfg.suffix
        self.savedir = {
            name:os.path.join(self.cfg.DATA_dir,name,'feature_all_anger'+self.suffix) for name in self.dataset_names
        }

        self.anger_frac = anger_frac
        self.GenAngerDataset()
    
    def GenAngerDataset(self):
        for name in self.dataset_names:
            Feature_Dir = self.raw_data_dir[name]
            with open(Feature_Dir,'rb') as rf:
                data = pickle.load(rf)
            
            for mode in self.modes:
                # get anger length
                self.angerDataset[name][mode] = dict()
                anger_idxs = np.arange(0,len(data[mode]['reg_lbls']),1)
                anger_idxs = anger_idxs[data[mode]['reg_lbls'].flatten()==-1]
                other_idxs = np.arange(0,len(data[mode]['reg_lbls']),1)
                other_idxs = other_idxs[data[mode]['reg_lbls'].flatten()>-1]
                
                anger_len = len(anger_idxs)
                other_len = int((anger_len/self.anger_frac)*(1-self.anger_frac))
                # get indexes
                np.random.shuffle(other_idxs)
                index_all = np.concatenate((anger_idxs,other_idxs[0:other_len]),axis=0)
                
                np.random.shuffle(index_all)
                
                for key in self.DatasetKeys:
                    if key=='num_samples':
                        self.angerDataset[name][mode][key] = anger_len+other_len
                    else:
                        self.angerDataset[name][mode][key] = data[mode][key][index_all]
                        print("name:{},mode:{},key:{},shape:{}".format(name,mode,key,self.angerDataset[name][mode][key].shape))
    def Save(self):
        for name in self.dataset_names:
            output_file = self.savedir[name] # which file to save？
            if os.path.exists(output_file):
                os.remove(output_file)            
            
            try:
                with open(output_file,'wb') as wf:
                    pickle.dump(self.angerDataset[name],wf)
                print('Lite features are saved at {}'.format(output_file))
            
            except:
                print('Cannot Save Feature at {}, cannot open target file.'.format(output_file))

if __name__=="__main__":
    frac = 0.5
    generator = AngerDatasetGenerator(frac)
    generator.Save()   
       