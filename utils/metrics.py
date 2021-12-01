#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   metrics.py
@Last Modified    :   2021/11/29 16:13:30
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

class Metrics():
    def __init__(self,mode):
        self.mode = mode
        assert self.mode in ['reg','cls']
        if self.mode =='reg':
            self.metrics = self.eval_regression
        elif self.mode =='cls':
            self.metrics = self.eval_classification
            
    def three_classifier(self,array_in,low_thres,high_thres):
        for i in range(len(array_in)):
            if array_in[i]<low_thres:
                array_in[i] = 0
            elif array_in[i]>=low_thres and array_in[i]<high_thres:
                array_in[i] = 1
            else:
                array_in[i] = 2
        
        return array_in
            
    def eval_regression(self,y_pred,y_true):
        
        # flatten
        preds = y_pred.view(-1).cpu().detach().numpy()
        truth = y_true.view(-1).cpu().detach().numpy()
        preds = np.clip(preds, a_min=-1., a_max=1.)
        
        num_samples = (len(preds))
        
        mae = np.mean(np.absolute(preds - truth))
        corr = np.corrcoef(preds, truth)[0][1]

        # 3 classes
        thres = [
            [-0.5,0.5],
            [-0.33,0.33],
            [-0.1,0.1]
        ]

        output_metrics = {}

        for idx,filter in enumerate(thres):
            # classify
            low_thres,high_thres = filter
            preds_cls = preds.copy()
            truth_cls = truth.copy()

            preds_cls = self.three_classifier(preds_cls,low_thres,high_thres)
            truth_cls = self.three_classifier(truth_cls,low_thres,high_thres)   

            f1score = f1_score(preds_cls,truth_cls,average='weighted')
            acc = accuracy_score(preds_cls,truth_cls)

            output_metrics['acc_{}'.format(high_thres)] = acc
            output_metrics['f1score_{}'.format(high_thres)] = f1score
        
        output_metrics['mae'] = mae
        output_metrics['corr'] = corr
        output_metrics['num_samples'] = num_samples
        return output_metrics

    def eval_classification(self,y_pred,y_true):
        preds = y_pred.view(-1).cpu().detach().numpy()
        truth = y_true.view(-1).cpu().detach().numpy()
        preds = np.clip(preds, a_min=-1., a_max=1.)

        mae = np.mean(np.absolute(preds - truth))
        corr = np.corrcoef(preds, truth)[0][1]


        output_metrics = {'mae':mae,'corr':corr}
        return output_metrics
