#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   metrics.py
@Last Modified    :   2021/11/29 16:13:30
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

from timeit import main
import torch
import numpy as np
import sys
sys.path.append('../')
import configure.config as cfg
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

class Metrics():
    def __init__(self,mode):
        self.mode = mode
        assert self.mode in ['reg','cls']
        if self.mode =='reg':
            self.metrics = self.eval_regression
        elif self.mode =='cls':
            self.metrics = self.eval_classification
        
        self.low_thres = cfg.LabelParas['low_thres']
        self.high_thres = cfg.LabelParas['high_thres']
        
    def three_classifier(self,array_in):
        for i in range(len(array_in)):
            if array_in[i]<=self.low_thres:
                array_in[i] = 0
            elif array_in[i]>self.low_thres and array_in[i]<=self.high_thres:
                array_in[i] = 1
            else:
                array_in[i] = 2
        
        return array_in
            
    def eval_regression(self,y_pred,y_true):
        '''
        args:
            y_pred : float between (-1,1)
            y_true : float between (-1,1)
            mode: 'reg','cls',
            low_thres,high_thres: 
                (-1,low_thres): negative
                (low_thres,high_thres): neutral
                (high_thres,1): positive
        '''
        
        # flatten
        preds = y_pred.view(-1).cpu().detach().numpy()
        truth = y_true.view(-1).cpu().detach().numpy()

        preds = np.clip(preds, a_min=-1., a_max=1.)
        num_samples = (len(preds))
        mae = np.mean(np.absolute(preds - truth))
        
        output_metrics = {}

        preds_cls = preds.copy()
        truth_cls = truth.copy()

        preds_cls = self.three_classifier(preds_cls)
        truth_cls = self.three_classifier(truth_cls)   

        f1score = f1_score(preds_cls,truth_cls,average='weighted')
        acc = accuracy_score(preds_cls,truth_cls)
        cm = confusion_matrix(truth,preds)

        output_metrics['acc'] = acc
        output_metrics['f1score'] = f1score
        
        output_metrics['mae'] = mae

        output_metrics['CM'] = cm
                    
        return output_metrics

    def eval_classification(self,y_pred,y_true):
        '''
        args:
            y_pred : num-classes-channel input e.g.:(0.1,0.8,0.1)
            y_true : one-hot encoding e.g.:(0,1,0)

        '''
        # to one dim
        
        preds = y_pred.view(-1).cpu().detach().numpy()
        truth = y_true.view(-1).cpu().detach().numpy()
        
    
        mae = np.mean(np.absolute(preds - truth))
        tiny = 1e-5
        if np.std(preds)==0:
            preds[0]+=tiny
        if np.std(truth)==0:
            truth[0]+=tiny
        #corr = np.corrcoef(preds, truth)[0][1]
        
        f1score = f1_score(preds,truth,average='weighted')
        acc = accuracy_score(preds,truth)
        
        cm = confusion_matrix(truth,preds)

        output_metrics = {'mae':mae,'f1score':f1score,'acc':acc,'CM':cm}
        return output_metrics

    def cls21hot(self,num_classes,y_in):
        '''
        transfer class label to one-hot encoding format
        y_in.shape: tensor:(length,1)
        y_out.shape: tensor:(length,num_classes)
        '''
        assert (max(y_in)==num_classes-1 and min(y_in)==0)
        y_out = np.zeros((y_in.shape[0],num_classes))
        for i in range(y_in.shape[0]):
            y_out[i][int(y_in[i])] = 1
        
        return torch.tensor(y_out)
    
    def onehot2cls(self,y_in):     
        '''
        transfer class label from one-hot encoding format to class format
        y_in.shape: tensor: (length,num_classes)
        y_out.shape: tensor: (length,1)
        '''
        assert (y_in.shape[-1]==3)
        return torch.argmax(y_in,dim=1)

        
if __name__ == '__main__':
    
    metrics = Metrics('cls')
    input = np.array([2,1,0,0,1,2])
    output = metrics.cls21hot(3,input)
    print(output)
    input = metrics.onehot2cls(output)
    print(input)