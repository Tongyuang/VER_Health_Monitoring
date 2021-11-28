#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   CUDAinit.py
@Last Modified    :   2021/11/28 20:22:33
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

import pynvml
import torch

def CUDA_Init():
    # CUDA Info obtained by pynvml
    available_gpus = list()
    pynvml.nvmlInit()
    
    target_gid, min_mem = 0,1e16
    DeviceCount = pynvml.nvmlDeviceGetCount()
    for idx in range(DeviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = meminfo.used
        # find the least used GPU
        if mem_used<min_mem:
            min_mem = mem_used
            target_gid = idx
            temp = pynvml.nvmlDeviceGetTemperature(handle,0)
    
    print("Using gpu {}, used memory:{},cur temperature:{} C".format(target_gid,min_mem,temp))
    available_gpus.append(target_gid)
    
    using_cuda = len(available_gpus)>0 and torch.cuda.is_available()
    
    device_name = 'cuda:%d'% int(available_gpus[0]) if using_cuda else 'cpu'
    device = torch.device(device_name)
    pynvml.nvmlShutdown()
    return device_name,device