import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from speakerDiarization import diarizeAudio
import gc
from pydub import AudioSegment
import time


def diarizeFromFolder(fromFolder,toFolder):
    INPUT_FOLDER_PATH = fromFolder
    OUTPUT_FOLDER_PATH = toFolder

    InputFiles = os.listdir(INPUT_FOLDER_PATH)
    print("All-Files:",InputFiles)
    # InputFiles = ["RID36_P104694_TMI_PTM_6104694_07172019_152726_001531_5.00.wav"]
    Total_time = 0
    total_audio_seconds = 0
    for ifile in InputFiles:
        print("Processing File:",ifile)
        file_name = ifile
        TOTAL_PATH = INPUT_FOLDER_PATH + file_name
        TOTAL_OUTPUT_PATH = OUTPUT_FOLDER_PATH + file_name.split(".")[0]+"/"
        if not os.path.exists(TOTAL_OUTPUT_PATH):
            os.makedirs(TOTAL_OUTPUT_PATH)

        audioSeconds = AudioSegment.from_file(TOTAL_PATH).duration_seconds
        start = time.time()
        diarizeAudio(TOTAL_PATH,TOTAL_OUTPUT_PATH,expectedSpeakers=2)
        end = time.time()

        computeTime = end-start
        computeSpeed = audioSeconds/computeTime
        print("Processing File Complete:",ifile)
        # print("Time Required for Computation:",(end-start)/60," minutes")
        # print("Computation Speed:",computeSpeed," s/s")

        Total_time += computeTime
        total_audio_seconds += audioSeconds
        
        collected = gc.collect()
        # print("Garbage Collector: Collected ",collected," objects")
    # print("Total Time Required for Process:",Total_time/60," minutes")
    # print("Total Audio-Time in the Process:",total_audio_seconds/60," minutes")
    # print("Average Compute Time:",total_audio_seconds/Total_time," s/s")



if __name__=="__main__":
    INPUT_FOLDER_PATH = "wavs/"
    OUTPUT_FOLDER_PATH = "Output/"
    diarizeFromFolder(INPUT_FOLDER_PATH,OUTPUT_FOLDER_PATH)

