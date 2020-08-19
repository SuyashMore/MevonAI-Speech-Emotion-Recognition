import numpy as np
import keras
import time
import librosa
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import sys
import librosa
import bulk_diarize as bk
from speakerDiarization import diarizeAudio
model = keras.models.load_model('model/lstm_cnn_rectangular_lowdropout_trainedoncustomdata.h5')

classes = ['Neutral', 'Happy', 'Sad',
           'Angry', 'Fearful', 'Disgusted', 'Surprised']


def predict(file, classes, model):
    solutions = []
    predictions = []
    # print(subdir,"+",file)
    temp = np.zeros((1,13,216))
    X, sample_rate = librosa.load(file, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    result = np.zeros((13,216))
    result[:mfccs.shape[0],:mfccs.shape[1]] = mfccs
    temp[0] = result
    t = np.expand_dims(temp,axis=3)
    ans=model.predict_classes(t)
    # print("SOL",classes[ans[0]])
    predictions.append(classes[ans[0]])

    if len(predictions) < 2:
        predictions.append('None') 

    solutions.append(predictions)
    return solutions[0][0]

# 
def computeEmotion(fileName):
    INPUT_FOLDER_PATH = fileName
    OUTPUT_FOLDER_PATH = "Output/"+fileName.split("/")[-1].split(".")[0]+"/"
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)


    predictions = []
    diazFiles = []
    originalFile = INPUT_FOLDER_PATH
    nUsers = diarizeAudio(INPUT_FOLDER_PATH,OUTPUT_FOLDER_PATH,expectedSpeakers=2)
    for i in range(0,nUsers):
        predictions.append(predict(OUTPUT_FOLDER_PATH+"_speaker"+str(i)+".wav", classes, model))
        diazFiles.append(OUTPUT_FOLDER_PATH+"_speaker"+str(i)+".wav")
    # print(predictions)

    return predictions,originalFile,diazFiles

if __name__ == '__main__':
    # INPUT_FOLDER_PATH = "data/Hindi/hello2.wav"
    # OUTPUT_FOLDER_PATH = "Output/"

    # # diarizeAudio(INPUT_FOLDER_PATH,OUTPUT_FOLDER_PATH,expectedSpeakers=2)
    # predictions = predict(OUTPUT_FOLDER_PATH+"_speaker0.wav", classes, model)
    # print(predictions)
    # predictions = predict(OUTPUT_FOLDER_PATH+"_speaker1.wav", classes, model)
    # print(predictions)

    print(computeEmotion("data/Hindi/hello2.wav"))


