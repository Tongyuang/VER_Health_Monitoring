from config import DataPreConfig,Emoconfig
import pickle
import shutil
import librosa
import argparse

class DataPreProcessor():
    def __init__(self,args):
        self.work_dir = args.work_dir
        self._DataPreConfig = DataPreConfig()

    def getAudioEmbeddings(self):# generate audio embedding
        for filedir in self._DataPreConfig.raw_wav_list:
            full_filedir = 
        
    