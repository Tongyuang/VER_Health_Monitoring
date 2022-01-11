#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   recorder.py
@Last Modified    :   2021/12/19 21:18:31
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib

import pyaudio
import time
import threading
import wave
import os
import shutil

class Recorder():
    def __init__(self, chunk=1024, channels=1, sr=22050):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.SR = sr
        self._running = True
        self._frames = []
        
    def start(self):
        threading._start_new_thread(self.__recording, ())
    def __recording(self):
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SR,
                input=True,
                frames_per_buffer=self.CHUNK)
        while(self._running):
            data = stream.read(self.CHUNK)
            self._frames.append(data)
  
        stream.stop_stream()
        stream.close()
        p.terminate()
  
    def stop(self):
        self._running = False
  
    def save(self, filename):
     
        p = pyaudio.PyAudio()
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.SR)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        print("Saved")
    

def start_recording(abs_root = './',usr_cases=10):
    
    name = str(input('please input your name:'))
    usr_root = os.path.abspath(os.path.join(abs_root,name))
    if os.path.exists(usr_root):
        shutil.rmtree(usr_root) 
    
    sex = str(input('male(m) or female(f)? m or f: '))
    while(sex not in ["m","f"]):
        sex = str(input('male(m) or female(f)? m or f: '))
    os.mkdir(usr_root)
    
    ret = [] # absolute dir
    i = 0
    while(i<usr_cases):
        rec = Recorder()
        
        num = str(input("press 1 to start:"))
        while(num!="1"):
           num = str(input("press 1 to start:"))
           
        tic = time.time()
        print("start recording...")
        rec.start()
        num  = str(input("press 2 to terminate:"))
        while(num!="2"):
            num = str(input("press 2 to terminate:"))     
        rec.stop()
        num = str(input("sure to save? y or n: "))
        while(num not in ["y","n"]):
           num = str(input("sure to save? y or n: "))
           
        if num=="y":
            toc = time.time()
            print("successfully get a record of length {:.4f} s.".format(toc-tic))
            output_file = "{}_{}.wav".format(name,i)
            rec.save(os.path.join(usr_root,output_file))
            ret.append(os.path.join(usr_root,output_file))
            i += 1
            
    return name,sex,ret

if __name__ == "__main__":

    ret = start_recording(usr_cases=12)
    print(ret)