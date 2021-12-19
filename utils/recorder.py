#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   recorder.py
@Last Modified    :   2021/12/19 21:18:31
@Author  :   Yuang Tong 
@Contact :   yuangtong1999@gmail.com
'''

# here put the import lib
import wave
import pyaudio
import time
 
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
SR = 22050
RECORD_SECONDS = 5
 
 
def record(filename='output.wav'):

  
    p = pyaudio.PyAudio()
  
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SR,
                    input=True,
                    frames_per_buffer=CHUNK)
  
    print("* recording")
  
    frames = []
 
    for i in range(0, int(SR / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
  
    print("* done recording")
  
    stream.stop_stream()
    stream.close()
    p.terminate()
  
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SR)
    wf.writeframes(b''.join(frames))
    wf.close()
 