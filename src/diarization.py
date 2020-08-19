from __future__ import print_function
import argparse
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioVisualization as aV
from pyAudioAnalysis import audioBasicIO
import scipy.io.wavfile as wavfile
import matplotlib.patches
from pydub import AudioSegment as pdb

speaker1_final = pdb.empty()
speaker2_final = pdb.empty()

op=None

def diarization_try(t1,t2,speakernumber):
    global speaker1_final, speaker2_final,op
    t1 = t1*1000
    t2 = t2*1000
    # op = aS.speakerDiarization(filename = "1.wav", n_speakers = 2, mt_size=2.0, mt_step=0.2, 
    #                    st_win=0.05, lda_dim=35, plot_res=False)
   
    Audio = "filtered.wav"
    if speakernumber == 0:
        speaker1 = pdb.from_wav(Audio)
        speaker1_final += speaker1[t1:t2]
    else:
        speaker2 = pdb.from_wav(Audio)
        speaker2_final += speaker2[t1:t2]
        
    
    


def timeStamps():
    global op, speaker1_final,speaker2_final
    prev=0
    last=0
    for i in range (len(op)):
        if op[i] != prev:
            diarization_try(0.2*last,0.2*(i-1),op[i]-1)
            print(0.2*last,":",0.2*i)
            last=i
        prev=op[i]    



if __name__ == "__main__":
    # global op, speaker1_final,speaker2_final 
    op= aS.speakerDiarization(filename = "filtered.wav", n_speakers = 2, mt_size=2.0, mt_step=0.2, 
                       st_win=0.05, lda_dim=35, plot_res=1)
    timeStamps()
    # diarization_try(0,10.42,0)
    # diarization_try(10.42,20.61,1)
    # diarization_try(20.61,30.85,0)
    # diarization_try(30.85,40.85,1)
    speaker1_final.export("speaker1.wav", format= "wav")
    speaker2_final.export("speaker2.wav", format= "wav")
