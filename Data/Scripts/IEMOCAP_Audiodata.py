'''
* @Author: Yuang Tong  
* @Date: 2021-11-24 14:15:19  
* @Last Modified by:   Yuang Tong  
* @Last Modified time: 2021-11-24 14:15:19 
''' 
import os
import argparse
import sys
sys.path.append('../../')

from configure.config import Emoconfig,IEMOCAP_dir,DataPreConfig

def GetAudioFileList(Dataset_dir):
    AudioFileList = list()
    LableFileList = list()

    sessions = []
    curdirs = os.listdir(Dataset_dir)
    
    for curdir in curdirs:
        if curdir.startswith('Session'):
            
            sessions.append(os.path.join(Dataset_dir,curdir))
    
    for sess in sessions:
        
        audio_piece_folder = os.path.join(sess,'sentences','wav')
        lbl_folder = os.path.join(sess,'dialog','EmoEvaluation')
        
        # audio
        AudioNameList = os.listdir(audio_piece_folder)
        for audioname in AudioNameList:
            if not audioname.startswith('.'):
                audionamedir = os.path.join(audio_piece_folder,audioname)
                for audio in os.listdir(audionamedir):
                    if (not audio.startswith('.') and audio.endswith('wav')):
                        AudioFileList.append(os.path.abspath(os.path.join(audionamedir,audio)))
        
        # lbl
        LblNameList = os.listdir(lbl_folder)
        for ff in LblNameList:
            if ff.endswith('txt') and (not ff.startswith('.')):
                LableFileList.append(os.path.join(lbl_folder,ff))
        
    AudioFileList.sort()
    LableFileList.sort()
    return (AudioFileList,LableFileList)

def read_txt(txt_file_dir):
    
    with open(txt_file_dir,"r") as f:
        ret = []
        for line in f.readlines():
            ret.append(line.strip('\n'))
    
    return ret

def GetAudioLabels(File_dir):
    names = list()
    labels = list()
    reg_lbls = list()
    lines = read_txt(File_dir)
    emoconfig = Emoconfig()
    for idx,line in enumerate(lines):
        if line.startswith('['):
            name = line.split('\t')[1]
            lbl = line.split('\t')[2]
            if lbl == 'xxx':
                lbl = emoconfig.AbbrevEmoDict[lines[idx+1].split('\t')[1].split(';')[0]]
        
            names.append(name)
            labels.append(lbl)
            reg_lbls.append(emoconfig.Annotation[lbl])
            
    
    return (names,labels,reg_lbls)

def GetFullAudioandLabels(AudioFileList,LabelFileList,output_dir = './test.txt'):
    FullNames = list()
    Fulllbls = list()
    FullRegLbls = list()
    for filename in LabelFileList:
        names,lbls,reg_lbls = GetAudioLabels(filename)
        FullNames += names
        Fulllbls += lbls
        FullRegLbls += reg_lbls
    assert(len(FullNames)==len(Fulllbls) and len(Fulllbls)==len(AudioFileList) and len(FullRegLbls)==len(FullNames))
    
    # output
    if os.path.exists(output_dir):
        os.remove(output_dir)

    writeseq = list()
    for idx in range(len(AudioFileList)):
        assert FullNames[idx] in AudioFileList[idx]
        writeseq.append("{}\t{}\t{}\t{}\n".format(FullNames[idx],AudioFileList[idx],Fulllbls[idx],FullRegLbls[idx]))
        
    with open(output_dir,"w") as f:
        f.writelines(writeseq)
        f.close()
    
    print('Total audio piece number: {}'.format(len(writeseq)))    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',type=str,default='/home/tongyuang/Dataset/VER/Dataset/IEMOCAP/Audio.txt',help='output file')
    args = parser.parse_args()

    config = DataPreConfig()
    IEMOCAP_dir = config.IEMOCAP_dir
    AudioFileList,LableFileList = GetAudioFileList(IEMOCAP_dir)
    GetFullAudioandLabels(AudioFileList,LableFileList,args.o)

    print('Audio data shortcut are stored at:{}'.format(args.o))