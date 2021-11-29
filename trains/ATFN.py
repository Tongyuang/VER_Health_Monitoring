import os
import time
import logging
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import sys
sys.path.append('../')
from configure.config import Model_ATFN_Config,Logger_Config
from utils.metrics import Metrics

import torch
import torch.nn as nn
from torch import optim


class ATFN():
    def __init__(self):
        self.model_config = Model_ATFN_Config()
        self.mode = self.model_config.mode

        self.criterion = nn.L1Loss() if self.mode=='reg' else nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            lr=self.model_config.ModelParas['learning_rate'])

        self.logger_config = Logger_Config()
        assert os.path.exists(self.logger_config.dir)
        self.logger_file_name = os.path.join(self.logger_config.dir,'{}.log'.format(
            time.strftime('%Y-%m-%d-%I-%M-%S', time.localtime())+'-ATFN'
        ))
        self.logger = logging
        self.logger.basicConfig(
                        level = self.logger_config.LEVEL,
                        format = self.logger_config.FORMAT,
                        filename = self.logger_file_name,
                        filemode = self.logger_config.FILEMODE
                    )
        
        self.metrics = Metrics(self.mode)

    def do_train(self,model,dataloader,device):
        epochs, best_epoch = 0, 0

        while(True):
            y_pred,y_true = list(),list()
            epochs += 1
            losses = []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    input = batch_data['feature']
                    lbl = batch_data['reg_lbl'] if self.mode=='reg' else batch_data['cls_lbl']
                    # to device
                    input = input.to(device)
                    lbl = lbl.to(device)
                    # clear gradient
                    self.optimizer.zero_grad()
                    # forward
                    outputs = model(input)
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

            train_loss = train_loss / len(dataloader['train'])
            self.logger.info("TRAINING: epoch:%d,best epoch:%d,loss: %.4f",epochs,best_epoch,train_loss)

            # evaluate
            pred, true = torch.cat(y_pred), torch.cat(y_true)

            train_results = self.Metrics.metrics(pred,true)
            output_str = ""
            for key in train_results.keys():
                output_str += key+":"+train_results[key]+" | "
            
            self.logger.info(output_str)

    def do_valid(self,model,dataloader,device):
        model.eval()
        y_pred,y_true = list(),list()
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader['valid']) as td:

                for batch_data in td:
                    input = batch_data['feature']
                    lbl = batch_data['reg_lbl'] if self.mode=='reg' else batch_data['cls_lbl']
                    # to device
                    input = input.to(device)
                    lbl = lbl.to(device)
                    # clear gradient
                    self.optimizer.zero_grad()
                    # forward
                    outputs = model(input)
                    # loss
                    loss = self.criterion(outputs,lbl)
                    # results
                    eval_loss += loss.item()
                    y_pred.append(outputs.cpu())
                    y_true.append(lbl.cpu())


            eval_loss = round(eval_loss / len(dataloader['valid']), 4)
            self.logger.info("EVALUATING: loss: %.4f",eval_loss)







