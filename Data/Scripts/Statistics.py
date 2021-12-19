#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Statistics.py
@Last Modified    :   2021/11/26 11:22:19
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

import sys

import pickle
import numpy  as np

sys.path.append('../../')

from configure.config import DataPreConfig

class Logger(object): # print on the screen and write in a file
    def __init__(self,filename):
        self.filename = filename
        self.terminal = sys.stdout
        self.log = open(self.filename, "a")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def GenStatistics():
    config = DataPreConfig()
    filename = config.feature_statistics_dir
    sys.stdout = Logger(filename)
    
    # write
    features = config.feature_store_dir
    for fea in features.keys():
        Feature_Dir = features[fea]
        with open(Feature_Dir,'rb') as rf:
            data = pickle.load(rf)
            
        modes = ['train','valid','test']

        print("DATASET {}".format(fea))
        for mod in modes:
            print("mode:",mod)
            for key in data[mod].keys():
                if (type(data[mod][key])==np.ndarray):
                    
                    print("key:{},shape:{}".format(key,data[mod][key].shape))
                    if key=='reg_lbls':
                        print("anger frac:{:.4f}".format(len(data[mod][key][data[mod][key]==-1])/len(data[mod][key])))
                else:
                    print("key:{},value:{}".format(key,data[mod][key]))
        print("-"*20)
        
if __name__ == '__main__':
    GenStatistics()
    
    
    

