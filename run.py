#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Last Modified    :   2021/11/26 11:08:19
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

import numpy as np
from torch.utils import data
from utils.CUDAinit import CUDA_Init
from utils.ModelUtils import count_parameters
from model.ModelSelector import ModelSelector
from trains.trainer import Trainer
from Data.Dataloader.AudioDataloader import AudioDataLoader

def start(model_name):
    # load device
    device_name,device = CUDA_Init()
    print("using device:{}".format(device_name))
    
    dataloader = AudioDataLoader()

    model = ModelSelector(model_name)
    model = model.to(device)
    print("This Model has {} trainable parameters".format(count_parameters(model)))
    
    trainer = Trainer(model,model_name,dataloader)
    trainer.do_train(device)

if __name__ == '__main__':
    start('ACN')
    

    


    
