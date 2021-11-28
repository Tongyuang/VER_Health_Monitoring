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
from utils.CUDAinit import CUDA_Init
from utils.ModelUtils import count_parameters
from model.ATFN import ATFN
from Data.Dataloader.AudioDataloader import AudioDataLoader

def start():
    # load device
    device_name,device = CUDA_Init()
    print("using device:{}".format(device_name))
    
    dataloader = AudioDataLoader()
    model = ATFN().to(device)
    
    print("This Model has {} trainable parameters".format(count_parameters(model)))

if __name__ == '__main__':
    start()
    

    


    
