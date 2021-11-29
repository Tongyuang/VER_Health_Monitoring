
import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

class Metrics():
    def __init__(self,mode):
        assert mode in ['reg','cls']
        if self.mode =='reg':
            self.metrics = self.eval_regression()
        elif self.mode =='cls':
            self.metrics = self.eval_classification()
    
    def eval_regression(self,y_pred,y_true):
        
        # flatten
        preds = y_pred.view(-1).cpu().detach().numpy()
        truth = y_true.view(-1).cpu().detach().numpy()
        preds = np.clip(preds, a_min=-1., a_max=1.)
        
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

            preds_cls[preds_cls<low_thres] = 0
            preds_cls[preds_cls>=low_thres and preds_cls<high_thres] = 1
            preds_cls[preds_cls>=high_thres] = 2
            
            truth_cls[truth_cls<low_thres] = 0
            truth_cls[truth_cls>=low_thres and truth_cls<high_thres] = 1
            truth_cls[truth_cls>=high_thres] = 2     

            f1score = f1_score(preds_cls,truth_cls,average='weight')
            acc = accuracy_score(preds,truth_cls)

            output_metrics['acc_{}'.format(high_thres)] = acc
            output_metrics['f1score_{}'.format(high_thres)] = f1score
        
        output_metrics['mae'] = mae
        output_metrics['corr'] = corr

        return output_metrics

    def eval_classification(self,y_pred,y_true):
        preds = y_pred.view(-1).cpu().detach().numpy()
        truth = y_true.view(-1).cpu().detach().numpy()
        preds = np.clip(preds, a_min=-1., a_max=1.)

        mae = np.mean(np.absolute(preds - truth))
        corr = np.corrcoef(preds, truth)[0][1]


        output_metrics = {'mae':mae,'corr':corr}
        return output_metrics
