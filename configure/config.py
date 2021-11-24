import shutil
import os



# Dataset Dir
CH_SIMS_dir = '/home/tongyuang/Dataset/VER/Dataset/CH_SIMS/Raw'
IEMOCAP_dir = '/home/tongyuang/Dataset/VER/Dataset/IEMOCAP/IEMOCAP_full_release'

class DataPreConfig():
    def __init__(self):
        self.dataset_names = ['IEMOCAP','CH_SIMS']
        self.raw_wav_list = {
            key: os.path.join(key,'Audio.txt') for key in self.dataset_names
        }
        # parameters when extracting features:
        self.hop_length = 512

        # parameters when padding:
        # MAXLEN = mean+3*std
        #self.Audio_MAXLEN = 100

        # split paras
        self.split_ratio = {
            "train":0.75,
            "valid":0.15,
            "test":0.10
        } 
        self.random_state = 1228

class Emoconfig():
    def __init__(self):
        self.Annotation = {
            "ang":-1,
            "sad":-0.6,
            "fea":-0.4,
            "fru":-0.2,
            "dis":-0.2,
            "neu":0,
            "hap":0.4,
            "sur":0.6,
            "exc":1.0,
            "oth":0
        }
        
        self.AbbrevEmoDict = {
            "Fear":"fea",
            "Frustration":"fru",
            "Neutral":"neu",
            "Anger":"ang",
            "Sadness":"sad",
            "Excited":"exc",
            "Happiness":"hap",
            "Surprise":"sur",
            "Other":"oth",
            "Disappointed":"fru"
        }


    def Reg2ClsLblCvtr(reg_lbl_in): # regress label -> class label
        if reg_lbl_in>0: # Positive
            return 2
        elif reg_lbl_in==0:  # Neutral
            return 1
        else:  # Negative
            return 0

        