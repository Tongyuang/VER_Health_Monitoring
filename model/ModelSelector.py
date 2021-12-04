#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ModelSelector.py
@Last Modified    :   2021/12/04 12:21:36
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

from .ACN import ACN
from .ALSTM import ALSTM
from .ATFN import ATFN

def ModelSelector(model_name):
    assert model_name in ['ACN','ALSTM','ATFN']
    if model_name=='ACN':
        model = ACN()
    elif model_name=='ALSTM':
        model = ALSTM()
    elif model_name =='ATFN':
        model = ATFN()
    
    return model