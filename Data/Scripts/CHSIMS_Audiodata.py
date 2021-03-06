#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   CHSIMS_Audiodata.py
@Time    :   2021/11/25 18:11:46
@Author  :   Yuang Tong 
@Version :   1.0
@Contact :   yuangtong1999@gmail.com
'''

import os
import argparse
import sys
sys.path.append('../../')

from configure.config import DataPreConfig

import pandas as pd

def GetAudioFileList(CH_SIMS_Dir):
    assert os.path.exists(CH_SIMS_Dir)
    AudioFileList = list()
    AudioFileName = list()
    Audio_Folder = os.path.join(CH_SIMS_Dir,'audio')
    for video_name in os.listdir(Audio_Folder):
        subpath = os.path.join(Audio_Folder,video_name)
        for audio in os.listdir(subpath):
            AudioFileName.append(video_name+'_'+audio.split('.')[0])
            AudioFileList.append(os.path.join(subpath,audio))
            
    # sort
    AudioFileList.sort()
    AudioFileName.sort()
    return AudioFileList,AudioFileName

def GetAudioLbls(CH_SIMS_Dir):
    assert os.path.exists(CH_SIMS_Dir)

    audio_lbl_dir = os.path.join(CH_SIMS_Dir,'SIMS-label.csv')
    audio_lbl = pd.read_csv(audio_lbl_dir)
    AudioCls = list(audio_lbl['annotation'].values)
    AudioRegLbls = list(audio_lbl['label_A'].values)
    
    return AudioRegLbls,AudioCls

def GenDescFile(CH_SIMS_Dir,output_file):
    if os.path.exists(output_file):
        print('re-writing at %s ...'%(output_file))
    
    assert output_file.endswith('.txt')
    
    DirList,NameList = GetAudioFileList(CH_SIMS_Dir)
    RegLbls, Cls = GetAudioLbls(CH_SIMS_Dir)
    
    writeseq = list()
    
    assert len(NameList)==len(RegLbls)
    
    for idx in range(len(NameList)):

        writeseq.append("{}\t{}\t{}\t{}\n".format(NameList[idx],DirList[idx],Cls[idx],RegLbls[idx]))
        
    with open(output_file,"w") as f:
        f.writelines(writeseq)
        f.close()
    

if __name__ == '__main__':
    
    config = DataPreConfig()
    CH_SIMS_Dir = config.CH_SIMS_dir
    output_file = '../AudioDir/CH_SIMS_Audio.txt'
    GenDescFile(CH_SIMS_Dir,output_file)
    print('Description File are stored at %s'%output_file)
    