'''
* @Author: Yuang Tong  
* @Date: 2021-11-24 18:41:00
* @Last Modified by:   Yuang Tong  
* @Last Modified time: 2021-11-24 18:41:00
''' 
import os
import argparse
import sys
sys.path.append('../')

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
    AudioRegLbls = list(audio_lbl['annotation'].values)
    AudioCls = list(audio_lbl['label_A'].values)
    
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
    
    CH_SIMS_Dir = '/home/tongyuang/Dataset/VER/Dataset/CH_SIMS/Raw'
    output_file = '/home/tongyuang/Dataset/VER/Dataset/CH_SIMS/Audio.txt'
    GenDescFile(CH_SIMS_Dir,output_file)
    print('Description File are stored at %s'%output_file)
    