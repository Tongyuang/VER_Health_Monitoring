#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ModelUtils.py
@Last Modified    :   2021/11/28 20:32:46
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

def count_parameters(model):
    answer = 0
    for p in model.parameters():
        if p.requires_grad:
            answer += p.numel()
                
    return answer