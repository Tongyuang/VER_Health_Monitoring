
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluate.py
@Last Modified    :   2021/11/30 10:22:14
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

import os
from numba.core.decorators import generated_jit
import numpy as np
import librosa
import sys
sys.path.append('../')

import time
from utils.CUDAinit import CUDA_Init
from model.ModelSelector import ModelSelector
from utils.metrics import Metrics
from utils.ModelUtils import count_parameters
from Data.Dataloader.AudioDataloader import AudioDataLoader
from utils import recorder
import configure.config as cfg
import torch


from Data.DataPre import DataPreProcessor

class Logger(object): # print on the screen and write in a file
    def __init__(self,filename):
        self.filename = filename
        self.terminal = sys.stdout
        self.log = open(self.filename, "a")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
    
class Evaluator():
    
    def __init__(self,model_save_dir,model_name):
        device_name,self.device = CUDA_Init()
        # model
        self.model = ModelSelector(model_name)
        self.model_name = model_name
        self.model.to(self.device)
        if self.model_name == 'ALSTM':
            self.model_config = cfg.Model_ALSTM_Config()
        elif self.model_name == 'ATFN':
            self.model_config = cfg.Model_ATFN_Config()
        elif self.model_name == 'ACN':
            self.model_config = cfg.Model_ACN_Config()

        self.mode = self.model_config.mode
        # load dict
        assert os.path.exists(model_save_dir)
        try:
            self.model.load_state_dict(torch.load(model_save_dir),strict=True)
        except:
            raise Exception("Cannot load model from {}, please check the model structure".format(model_save_dir))
        
        print("This Model has {} parameters".format(count_parameters(self.model)))
    def forward(self,input):
        '''
        input: feature of shape(batch_size, feature_dim)
        output: (batch_size,1) if reg mode, (batch_size,num_classes) if cls mode
                inference time (in sec)
        '''
        self.model.eval()
        input = torch.tensor(input)
        input = input.to(self.device,dtype=torch.float32)
        input = input.view((input.shape[0],1,input.shape[-1])) # (batch_size,1,feature_dim)
        
        tic = time.time()
        output = self.model(input)
        output = output.cpu()
        toc = time.time()
        return output,(toc-tic)
    
def preprocessor(wav_dir_list,feature_dim=562):
    '''
    args: wav_dir_list: list of absolute wav dir
    '''
    output = np.zeros((len(wav_dir_list),feature_dim))
    dpp = DataPreProcessor()
    for i,wav_dir in enumerate(wav_dir_list):
        assert os.path.exists(wav_dir)
        feature_out = np.mean(dpp.getSingleAudioEmbd(wav_dir),axis=0)
        output[i] = feature_out
    return output


def get_pred_results(model_output):
    '''
    args:
        input: model_output: torch.tensor(batch_size,num_classes)
        output: anger percentage
    '''       
    model_output.detach().numpy()
    ret = np.zeros((model_output.shape[0],2))
    
    for i in range(model_output.shape[0]):
        print(model_output[i][0],model_output[i][1])
        ret[i][0] = model_output[i][0]/(model_output[i][0]+model_output[i][1]) # prob
        ret[i][1] = model_output[i][0]+model_output[i][1] # activation level
    return ret
    
def re_normalize(results):
    '''
    results[0,:] anger probability
    results[1,:] anger activation level
    '''
    low_thres = np.mean(results,axis=0)[0]-1.5*np.std(results,axis=0)[0]
    for i in range(results.shape[0]):
        results[i][0] = (results[i][0]-low_thres)/(1-low_thres)
        if results[i][0]<0:
            results[i][0] = 0.5*np.random.rand()

if __name__ == '__main__':

    
    #name,sex,wav_list = recorder.start_recording(abs_root='./usr_cases/',usr_cases=12)
    #filename = os.path.join('./usr_cases/',name,'statistics.txt')

    
    '''
    name = "ziqi"
    sex = "m"
    wav_list = ["./usr_cases/{}/{}_{}.wav".format(name,name,i) for i in range(12)]
    filename = "./usr_cases/{}/statistics.txt".format(name)
    sys.stdout = Logger(filename)
    '''
    wav_list = ["./test/record.wav"]
    sex = "m"
    print("analysing...")
    input = preprocessor(wav_list)
    
    model_save_dir = "../results/best_models/2021-12-20-03-52-41-ACN-cls-best.pt"
    model_name = "ACN"
    
    evaluator = Evaluator(model_save_dir=model_save_dir,
                          model_name = model_name)
    
    output,time_interval = evaluator.forward(input)
    results = get_pred_results(output)
    if sex=="f":
        re_normalize(results)
    for i in range(len(results)):

        print("wav_{},anger probability: {:.4f},activation level:{:.4f}".format(i,results[i][0],results[i][1]))
    
    print("evaluation time: {:.4f} ms".format(time_interval*1000))

