import shutil


class DataPreConfig():
    def __init__(self):
        self.dataset_names = ['IEMOCAP','CH_SIMS']
        self.raw_wav_list = {
            key: 'dataset/'+key+'/Audio.txt' for key in self.dataset_names
        }

class Emoconfig():
    def __init__(self):
        self.Annotation = {
            "ang":-1,
            "sad":-0.6,
            "fear":-0.4,
            "fru":-0.2,
            "neu":0,
            "hap":0.4,
            "sur":0.6,
            "exc":1.0,
            "oth":0
        }
        
        self.AbbrevEmoDict = {
            "fear":"fea",
            "frustration":"fru",
            "neutral":"neu",
            "anger":"ang",
            "sadness":"sad",
            "excited":"exc",
            "happiness":"hap",
            "surprise":"sur",
            "other":"oth",
            "disappointed":"fru"
        }


    def Reg2ClsLblCvtr(reg_lbl_in): # regress label -> class label
        if reg_lbl_in>0: # Positive
            return 2
        elif reg_lbl_in==0:  # Neutral
            return 1
        else:  # Negative
            return 0

        