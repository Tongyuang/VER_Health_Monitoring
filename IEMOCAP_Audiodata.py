import os
import argparse

def GetAudioFileList(Dataset_dir='./Dataset/IEMOCAP/IEMOCAP_full_release'):
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
    lines = read_txt(File_dir)
    for idx,line in enumerate(lines):
        if line.startswith('['):
            name = line.split('\t')[1]
            lbl = line.split('\t')[2]
            if lbl == 'xxx':
                lbl = abbrev_for_emo(lines[idx+1].split('\t')[1].split(';')[0])
        
            names.append(name)
            labels.append(lbl)
    
    return (names,labels)

def GetFullAudioandLabels(AudioFileList,LabelFileList,output_dir = './test.txt'):
    FullNames = list()
    Fulllbls = list()

    for filename in LabelFileList:
        names,lbls = GetAudioLabels(filename)
        FullNames += names
        Fulllbls += lbls
    
    assert(len(FullNames)==len(Fulllbls) and len(Fulllbls)==len(AudioFileList))
    
    # output
    if os.path.exists(output_dir):
        os.remove(output_dir)

    writeseq = list()
    for idx in range(len(AudioFileList)):
        assert FullNames[idx] in AudioFileList[idx]
        writeseq.append("{}\t{}\t{}\n".format(FullNames[idx],AudioFileList[idx],Fulllbls[idx]))
        
    with open(output_dir,"w") as f:
        f.writelines(writeseq)
        f.close()
    
    print('Total audio piece number: {}'.format(len(writeseq)))    
    
def abbrev_for_emo(emo_in):
    if emo_in in ['Frustration','frustration']:
        return 'fru'
    if emo_in in ['Neutral','neutral']:
        return 'neu'
    if emo_in in ['Anger','anger']:
        return 'ang'
    if emo_in in ['Other','other']:
        return 'neu'
    if emo_in in ['Sadness','sadness']:
        return 'sad'
    if emo_in in ['Excited','excited']:
        return 'exc'
    if emo_in in ['Happiness','happiness']:
        return 'hap'
    if emo_in in ['Surprise','surprise']:
        return 'sup'
    return emo_in

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir',type=str,default='./Dataset/IEMOCAP/IEMOCAP_full_release',help='dataset dir')
    parser.add_argument('-o',type=str,default='./test.txt',help='dataset dir')
    args = parser.parse_args()

    AudioFileList,LableFileList = GetAudioFileList(args.dir)
    GetFullAudioandLabels(AudioFileList,LableFileList,args.o)

    print('Audio data shortcut are stored at:{}'.format(args.o))