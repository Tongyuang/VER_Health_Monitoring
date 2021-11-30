#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ATFN.py
@Last Modified    :   2021/11/29 16:04:51
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

import os
import time
import logging
import argparse
import numpy as np
from glob import glob
from scipy.sparse import data
from tqdm import tqdm

import sys
sys.path.append('../')
from configure.config import Model_ATFN_Config,Logger_Config,DataPreConfig
from utils.metrics import Metrics

import torch
import torch.nn as nn
from torch import optim
from torch.utils.mobile_optimizer import optimize_for_mobile


class ATFN_Trainer():
    def __init__(self,model,dataloader):

        self.dataloader = dataloader
        self.model = model
        self.model_config = Model_ATFN_Config()
        self.mode = self.model_config.mode
        
        self.datapre_config = DataPreConfig()
        self.workdir = self.datapre_config.WORK_dir

        self.criterion = nn.L1Loss() if self.mode=='reg' else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.model.parameters(),lr=self.model_config.ModelParas['learning_rate'])

        self.logger_config = Logger_Config()
        assert os.path.exists(self.logger_config.dir)
        self.logger_file_name = os.path.join(self.logger_config.dir,'{}.log'.format( \
            time.strftime('%Y-%m-%d-%I-%M-%S', time.localtime())+'-ATFN' \
        ))
        
        model_save_dir = os.path.join(self.workdir,'results','best_models')
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        self.model_save_path = os.path.join(model_save_dir,'{}.pt'.format( \
            time.strftime('%Y-%m-%d-%I-%M-%S', time.localtime())+'-ATFN-best' \
        ))
        
        if self.model_config.commonParas['gen_lite_model_for_mobile']:
            # save lite model
            lite_model_save_dir = os.path.join(self.workdir,'results','pub_models')
            self.lite_model_save_path = os.path.join(lite_model_save_dir,'{}.ptl'.format( \
            time.strftime('%Y-%m-%d-%I-%M-%S', time.localtime())+'-ATFN-best' \
        ))

        self.logger = logging
        self.logger.basicConfig(
                        level = self.logger_config.LEVEL,
                        format = self.logger_config.FORMAT,
                        filename = self.logger_file_name,
                        filemode = self.logger_config.FILEMODE
                    )
        
        self.metrics = Metrics(self.mode)

        self.early_stop = self.model_config.commonParas['early_stop']
        
    def do_train(self,device):
        epochs, best_epoch = 0, 0
        min_eval_loss = 1e6
        while(True):
            y_pred,y_true = list(),list()
            epochs += 1
            losses = []
            self.model.train()
            train_loss = 0.0
            with tqdm(self.dataloader['train']) as td:
                for batch_data in td:
                    input = batch_data['feature']
                    lbl = batch_data['reg_lbl'] if self.mode=='reg' else batch_data['cls_lbl']
                    # to device
                    input = input.to(device)
                    lbl = lbl.to(device)
                    # clear gradient
                    self.optimizer.zero_grad()
                    # forward
                    outputs = self.model(input)
                    # loss
                    loss = self.criterion(outputs,lbl)

                    # update
                    loss.backward()
                    self.optimizer.step()
                    # results
                    train_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(lbl.cpu())
                    losses.append(train_loss)

            train_loss = train_loss / len(self.dataloader['train'])
            self.logger.info("TRAINING: epoch:%d, best epoch:%d,loss: %.4f",epochs,best_epoch,train_loss)

            # evaluate
            pred, true = torch.cat(y_pred), torch.cat(y_true)

            train_results = self.metrics.metrics(pred,true)
            output_str = ""
            for key in train_results.keys():
                output_str += ("{key}:{val:4f} |".format(key=key,val=train_results[key]))
            self.logger.info(output_str)

            eval_loss = self.do_valid(device)
            # save the best model
            if eval_loss < min_eval_loss- 1e-5: # only updates when there's a significance difference
                best_epoch, min_eval_loss = epochs,eval_loss
                self.logger.info("Best epoch:%d, min val loss: %.4f",best_epoch,min_eval_loss)
                # save
                torch.save(self.model.cpu().state_dict(),self.model_save_path)
                self.logger.info("Best Model are saved at %s",self.model_save_path)
                self.model.to(device)
            
            if epochs - best_epoch >= self.early_stop:
                
                print("early stop triggered!")
                self.logger.info("early stop triggered!")
                self.logger.info("Best Model are saved at %s",self.model_save_path)
                # generate lite model for mobile
                if self.model_config.commonParas['gen_lite_model_for_mobile']:
                    quantized_model = torch.quantization.quantize_dynamic( \
                        self.model.to('cpu'), qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
                    scripted_model = torch.jit.script(quantized_model)
                    optimized_model = optimize_for_mobile(scripted_model)
                    optimized_model._save_for_lite_interpreter(self.lite_model_save_path)

                print("finished.")
                return


    def do_valid(self,device):
        self.model.eval()
        y_pred,y_true = list(),list()
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(self.dataloader['valid']) as td:

                for batch_data in td:
                    input = batch_data['feature']
                    lbl = batch_data['reg_lbl'] if self.mode=='reg' else batch_data['cls_lbl']
                    # to device
                    input = input.to(device)
                    lbl = lbl.to(device)
                    # clear gradient
                    self.optimizer.zero_grad()
                    # forward
                    outputs = self.model(input)
                    # loss
                    loss = self.criterion(outputs,lbl)
                    # results
                    eval_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(lbl.cpu())


            eval_loss = round(eval_loss / len(self.dataloader['valid']), 4)
            self.logger.info("EVALUATING: loss: %.4f",eval_loss)

            # evaluate
            pred, true = torch.cat(y_pred), torch.cat(y_true)

            valid_results = self.metrics.metrics(pred,true)
            output_str = ""
            for key in valid_results.keys():
                output_str += ("{key}:{val:4f} |".format(key=key,val=valid_results[key]))
           
            self.logger.info(output_str)
            return eval_loss






