#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DataPre.py
@Last Modified    :   2021/11/25 18:12:26
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

import sys 
sys.path.append("../") 

from numpy.typing import _256Bit
from configure.config import DataPreConfig,Emoconfig
import os
import pickle
import librosa
import argparse
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split

from tqdm import tqdm

class DataPreProcessor():
    def __init__(self,args):
        self.work_dir = args.work_dir
        self.data_dir = args.data_dir
        self._DataPreConfig = DataPreConfig()
        self._Emoconfig = Emoconfig()
        self.padding_mode = args.padding_mode
        self.padding_loc = args.padding_loc
        

    def getSingleAudioEmbd(self,wav_dir): # generate single audio embedding
        assert os.path.exists(wav_dir)
        y,sr = librosa.load(wav_dir)
        hop_length = self._DataPreConfig.hop_length
        f0 = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).T # (seq_len, 1)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, htk=True).T # (seq_len, 20)
        cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T # (seq_len, 12)
        return np.concatenate([f0,mfcc,cqt],axis=-1) #(seqlen,33)

    def padding(self,feature,MAX_LEN):
        # input: (seqlen,feature_dim)
        # output: (MAXLEN,feature_dim)
        assert self.padding_mode in ['zeros','normal']
        assert self.padding_loc in ['front','back']

        length = feature.shape[0]
        
        if length > MAX_LEN:
            # cut
            start = int((length-MAX_LEN)/2)
            end = start+MAX_LEN
            return feature[start:end,:]
        
        elif length == MAX_LEN:
            return feature
        
        pad_len = MAX_LEN-length
        if self.padding_mode == "zeros":
            pad = np.zeros([pad_len,feature.shape[-1]])
        else:
            mean,std = feature.mean(),feature.std()
            pad = np.random.normal(mean,std,(pad_len,feature.shape[-1]))
            
        feature = np.concatenate([pad, feature], axis=0) if(self.padding_loc == "front") else \
                  np.concatenate((feature, pad), axis=0)
        
        return feature

    def getAudioEmbeddings(self):# generate full audio embedding
        self.AudioEmbeddings = {name:{} for name in self._DataPreConfig.dataset_names}
        for name in self._DataPreConfig.dataset_names:
            filedir = self._DataPreConfig.raw_wav_list[name]
            self.AudioEmbeddings[name] = {
                'feature':[],
                'lengths':[],
                'reg_lbl':[]
            }

            full_filedir = os.path.join(self.work_dir,filedir)
            with open(full_filedir,"r") as f:
                print('start calculating feature of %s'%name)
                for line in tqdm(f.readlines()):
                    (wav_name,wav_dir,_,reg_lbl) = line.split('\t')

                    # load data
                    feature = self.getSingleAudioEmbd(wav_dir)
                    self.AudioEmbeddings[name]['feature'].append(feature)
                    self.AudioEmbeddings[name]['lengths'].append(len(feature))
                    self.AudioEmbeddings[name]['reg_lbl'].append(reg_lbl)
        
    def getFinalAudioEmbedding(self):
        # get final embedding
        self.getAudioEmbeddings()
        self.split_mode = ['train','valid','test']
        
        self.FinalAudioEmbeddings = {name:{} for name in self.AudioEmbeddings.keys()}

        for name in self.AudioEmbeddings.keys():
            # final audio embedding

            # padding
            lengths = self.AudioEmbeddings[name]['lengths']
            mean,std = np.mean(lengths),np.std(lengths)

            # final feature
            final_feature_lengths = int(mean+3*std)
            num_samples = len(lengths)
            feature_dim = len(self.AudioEmbeddings[name]['feature'][0][-1])

            feature = np.zeros((num_samples,final_feature_lengths,feature_dim))
            reg_lbls = np.zeros((num_samples,1))

            print('start calculating output feature of %s'%name)
            
            for idx in tqdm(range(len(self.AudioEmbeddings[name]['feature']))):
                # feature
                fea = self.AudioEmbeddings[name]['feature'][idx]
 
                feature[idx] = self.padding(fea,final_feature_lengths)
                
                # lbl
                reg_lbl = self.AudioEmbeddings[name]['reg_lbl'][idx]
                reg_lbls[idx] = reg_lbl
            
            # split
            feature_vi,feature_test,lbl_vi,lbl_test = train_test_split(
                feature,
                reg_lbls,
                test_size=self._DataPreConfig.split_ratio["test"],
                random_state=self._DataPreConfig.random_state
            )

            feature_train,feature_valid,lbl_train,lbl_valid = train_test_split(
                feature_vi,
                lbl_vi,
                test_size = self._DataPreConfig.split_ratio["valid"]/(1-self._DataPreConfig.split_ratio["test"]),
                random_state=self._DataPreConfig.random_state
            )
            feature_dict = {'train':feature_train,'valid':feature_valid,'test':feature_test}
            lbl_dict = {'train':lbl_train,'valid':lbl_valid,'test':lbl_test}

            # to FinalAudioEmbeddings
            self.FinalAudioEmbeddings[name] = {mode:{} for mode in self.split_mode}

            for mode in self.split_mode:
                cur_feature = feature_dict[mode]
                cur_reg_lbl = lbl_dict[mode]
                num_samples = len(cur_reg_lbl)
                # to finalaudioembeddings
                self.FinalAudioEmbeddings[name][mode]['feature'] =  cur_feature
                self.FinalAudioEmbeddings[name][mode]['num_samples'] = num_samples
                self.FinalAudioEmbeddings[name][mode]['reg_lbls'] = cur_reg_lbl
                self.FinalAudioEmbeddings[name][mode]['cls_lbls'] = self.Reg2ClsCvtr(cur_reg_lbl)

            
    def Reg2ClsCvtr(self,reg_lbls_in):
        assert reg_lbls_in.shape[-1]==1
        cls_lbls = np.zeros((reg_lbls_in.shape[0],1))
        for idx in range(len(reg_lbls_in)):
            cls_lbls[idx] = self._Emoconfig.Reg2ClsLblCvtr(reg_lbls_in[idx][0])
        return cls_lbls

    def SaveFeature(self):
        self.getFinalAudioEmbedding()
        for name in self.FinalAudioEmbeddings.keys():
            suffix = '.pkl'
            output_dir = os.path.join(self.data_dir,name)
            output_file = os.path.join(output_dir,'feature'+suffix)
            if os.path.exists(output_file):
                os.remove(output_file)
            
            try:
                with open(output_file,'wb') as wf:
                    pickle.dump(self.FinalAudioEmbeddings[name],wf)
                print('Features are saved in %s .'%output_file)
            except:
                print("Cannot Save Features at {}".format(output_file))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--work_dir', type=str, default='../',
                        help='work dir')
    parser.add_argument('--data_dir',type=str,default='/home/tongyuang/Dataset/VER/Dataset/')
    parser.add_argument('--padding_mode', type=str, default="zeros",
                        help='support [\'zeros\',\'normal\']')
    parser.add_argument('--padding_loc', type=str, default="back",
                        help='support [\'front\',\'back\']')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    datapreprocessor = DataPreProcessor(args)
    datapreprocessor.SaveFeature()

    