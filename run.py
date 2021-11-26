#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Last Modified    :   2021/11/26 11:08:19
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

import pickle
import numpy as np

Feature_Dir = '/home/tongyuang/Dataset/VER/Dataset/CH_SIMS/feature.pkl'

with open(Feature_Dir,'rb') as rf:
    data = pickle.load(rf)

modes = ['train','valid','test']

for mod in modes:
    print("mode:",mod)
    for key in data[mod].keys():
        print("key:{},shape:{}".format(key,data[mod][key].shape) if type(data[mod][key])==np.ndarray else \
            "key:{},value:{}".format(key,data[mod][key]))
    
