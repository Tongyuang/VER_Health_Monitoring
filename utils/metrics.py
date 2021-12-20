#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   metrics.py
@Last Modified    :   2021/12/19 18:00:42
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
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
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
    
    def two_classifier(self,array_in):
        for i in range(len(array_in)):
            if array_in[i]<=self.low_thres:
                array_in[i] = 0
            else:
                array_in[i] = 1
        
        return array_in
            
    def eval_regression(self,y_pred,y_true):
        '''
        args:
            y_pred : float between (-1,1)
            y_true : float between (-1,1)
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
            y_pred : class preds. e.g.: [0,0,1,1,2,2,3,3,]......
            y_true : class preds. e.g.: [0,0,1,1,2,2,3,3,]......

        '''
        # to one dim
        
        preds = y_pred.view(-1).cpu().detach().numpy()
        truth = y_true.view(-1).cpu().detach().numpy()
        
    
        mae = np.mean(np.absolute(preds - truth))
        
        
        acc = accuracy_score(truth,preds)
        prec = precision_score(truth,preds)
        recall = recall_score(truth,preds)
        f1score = f1_score(truth,preds,average='weighted')
        cm = confusion_matrix(truth,preds)

        output_metrics = {'mae':mae,'f1score':f1score,'acc':acc,'prec':prec,'rec':recall,'CM':cm}
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
        return torch.argmax(y_in,dim=1)

        
if __name__ == '__main__':
    
    metrics = Metrics('cls')
    input = np.array([2,1,0,0,1,2])
    output = metrics.cls21hot(3,input)
    print(output)
    input = metrics.onehot2cls(output)
    print(input)