#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Last Modified    :   2021/12/03 17:13:41
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib


import shutil
import os
import logging


LabelParas = {
    'low_thres':-1,
    'high_thres':-1
    
}

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
            name:os.path.join(self.DATA_dir,name,'feature_all_anger'+self.suffix) for name in self.dataset_names
        }      
        # feature_statistics
        self.feature_statistics_dir = os.path.join(self.WORK_dir,'Data','Scripts','Statistics_All.txt')

        self.num_classes = 2
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


    def Reg2ClsLblCvtr(self,low_thres,high_thres,reg_lbl_in): # regress label -> class label
        if reg_lbl_in>high_thres: # Positive
            return 2
        elif reg_lbl_in<=low_thres:  # Neutral
            return 0
        else:  # Negative
            return 1

class DataLoaderConfig():
    def __init__(self):
        
        self.BatchSize = 32
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
            'feature_dim':562,
            
            'hidden_dims' : [512,256,128,32,16],
            'hidden_dim': 32,
            'dropout': 0.2,
            
            'post_hidden_dim':4,
            'post_dropout':0.2,
            
            'num_classes' :2,
            
            'activation':'leaky_relu', # must in ['relu','leaky_relu','tanh']
            'output_activation':'tanh', # must in ['relu','leaky_relu','tanh']
            'output_activation_for_classification':'sigmoid',
            
            'learning_rate': 5e-3,
            'weight_decay': 1e-4,
        }
        

class Model_ACN_Config():
    def __init__(self):
        self.mode = 'cls'
        
        self.commonParas = {
            'early_stop': 16,
            'gen_lite_model_for_mobile':True,
        }
        
        self.ModelParas = {
            'feature_dim':562,
            'num_classes' :2,
            'activation':'relu',# must in ['relu','leaky_relu','tanh']
                        
            'channels' :[16,64,64,128,128,256,256], # 
            'kernels': [7,7,7,5,5,3,3],# must odd number
            'pooling_kernels' :[1,1,4,1,4,1,4], # 1 means no pooling
            'dropout': [0.0,0.0,0.0,0.0,0.0,0.2,0.2],
            # output
            'output_hidden_dim':8*256, # 
            'output_dropout': 0.2,
            'output_activation':'tanh', # must in ['relu','leaky_relu','tanh']
            'output_activation_for_classification':'relu',
            
            'learning_rate': 1e-5,
            'weight_decay': 1e-4,
            
        }
        
class Model_ALSTM_Config():
    def __init__(self):
        self.mode = 'reg'
        
        self.commonParas = {
            'early_stop': 8,
            'gen_lite_model_for_mobile':True,
        }
        self.ModelParas = {
            'num_classes':2,
            'feature_dim':33,
            'sequence_length': 588, # remember to check the './Data/Scripts/Statistics.txt'
            
            'dropout':0.2,
            'hidden_size':32,
            'num_layers':64,
            
            'norm_mode':'last', # how to normalize the result, must in ['last','mean']
            
            'linear_hidden_dim':16, # the hidden dim of linear output layer
            'output_activation':'tanh',# must in ['relu','leaky_relu','tanh']
            'output_activation_for_classification':'relu',
            
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
        