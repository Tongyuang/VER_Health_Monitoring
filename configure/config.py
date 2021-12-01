#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Last Modified    :   2021/11/26 14:58:49
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib


import shutil
import os
import logging


class DataPreConfig():
    def __init__(self):
        
        self.WORK_dir = '/home/tongyuang/code/VER_Health_Monitoring'
        
        # Dataset Dir
        self.DATA_dir = '/home/tongyuang/Dataset/VER/Dataset/'
        self.CH_SIMS_dir = '/home/tongyuang/Dataset/VER/Dataset/CH_SIMS/Raw'
        self.IEMOCAP_dir = '/home/tongyuang/Dataset/VER/Dataset/IEMOCAP/IEMOCAP_full_release'
        
        self.dataset_names = ['IEMOCAP','CH_SIMS']
        self.raw_wav_list = {
            key: os.path.join(self.WORK_dir,'Data','AudioDir',key+'_Audio.txt') for key in self.dataset_names
        }
        
        
        # parameters when extracting features:
        self.hop_length = 512
        
        # parameters when padding:
        self.padding_mode = "zeros" #["zeros","normal"]
        self.padding_loc = "front" # ["front","back"]
        

        # split paras
        self.split_ratio = {
            "train":0.75,
            "valid":0.15,
            "test":0.10
        } 
        self.random_state = 1228
        
        # output feature
        self.suffix = '.pkl'       
        self.feature_store_dir = {
            name:os.path.join(self.DATA_dir,name,'feature'+self.suffix) for name in self.dataset_names
        }      
        # feature_statistics
        self.feature_statistics_dir = os.path.join(self.WORK_dir,'Data','Scripts','Statistics.txt')

class Emoconfig():
    def __init__(self):
        self.Annotation = {
            "ang":-1,
            "sad":-0.6,
            "fea":-0.4,
            "fru":-0.2,
            "dis":-0.2,
            "neu":0,
            "hap":0.4,
            "sur":0.6,
            "exc":1.0,
            "oth":0
        }
        
        self.AbbrevEmoDict = {
            "Fear":"fea",
            "Frustration":"fru",
            "Neutral":"neu",
            "Anger":"ang",
            "Sadness":"sad",
            "Excited":"exc",
            "Happiness":"hap",
            "Surprise":"sur",
            "Other":"oth",
            "Disappointed":"fru"
        }


    def Reg2ClsLblCvtr(self,reg_lbl_in): # regress label -> class label
        if reg_lbl_in>0: # Positive
            return 2
        elif reg_lbl_in==0:  # Neutral
            return 1
        else:  # Negative
            return 0

class DataLoaderConfig():
    def __init__(self):
        
        self.BatchSize = 128
        self.num_worker = 8

class Model_ATFN_Config():
    def __init__(self):
        
        self.mode = 'reg' # must in ['reg','cls']
        
        self.SubnetParas = {
            'activation':'leaky_relu',# must in ['relu','leaky_relu','tanh']
        }
        self.commonParas = {
            'early_stop': 8,
            'gen_lite_model_for_mobile':True,
        }
        
        self.ModelParas = {
            'feature_dim':33,
            'hidden_dim': 32,
            'dropout': 0.2,
            
            'post_hidden_dim':4,
            'post_drouput':0.2,
            
            'num_classes' :3,
            
            'activation':'leaky_relu', # must in ['relu','leaky_relu','tanh']
            'output_activation':'tanh', # must in ['relu','leaky_relu','tanh']
            
            'learning_rate': 5e-3,
            'weight_decay': 1e-4,
        }

class Model_ACN_Config():
    def __init__(self):
        self.mode = 'reg'
        
        self.commonParas = {
            'early_stop': 8,
            'gen_lite_model_for_mobile':True,
        }
        
        self.ModelParas = {
            'feature_dim':33,
            
            'channels' :[16,32,64],

            'kernel_size': 3,
            'pooling_kernel_size':2,
            
            'output_hidden_dim':64*4,
            
            'dropout': 0.2,
            
            'num_classes' :3,
            
            'activation':'relu',# must in ['relu','leaky_relu','tanh']
            'output_activation':'tanh', # must in ['relu','leaky_relu','tanh']
            
            'learning_rate': 5e-4,
            'weight_decay': 1e-4,
            
        }
        

class Logger_Config():
    def __init__(self):
        self.DataPreConfig = DataPreConfig()
        self.FORMAT = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        self.LEVEL = logging.DEBUG
        self.dir = os.path.join(self.DataPreConfig.WORK_dir,'results','logs')

        self.FILEMODE = 'a'
        