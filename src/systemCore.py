import Mega_Submission as Emotion
import gcloud
import ibmWatson
import database

def deepAnalysis(audioFile):
    print("Audio Analysis")
    predictions , original_file,diaz_files=Emotion.computeEmotion(audioFile)
    print("Audio Analysis Complete")
    print("Predictions:",predictions)
    print("Speech to Text:(Transcripting)")
    transcript = gcloud.recognizeSpeech(audioFile)
    print("Transcripting Complete")
    print("Trancript:",transcript)
    print("Transcript Analysis...")
    emotion,sentiment,category=ibmWatson.analyzeText(transcript)
    print("Transcript Analysis Complete")
    call_duration = gcloud.getAudioLength(audioFile)

    return {"callDuration":call_duration,
            "predictions":predictions,
            "diaz_files":diaz_files,
            "transcript":transcript,
            "emotion":emotion,
            "sentiment":sentiment,
            "category":category}



if __name__=="__main__":
    
    # print(gcloud.getAudioLength("data/ll1.wav"))
    response = (deepAnalysis("data/ll1.wav"))
    database.appendData("Ajay1667",response)
    # print("Publishing to Database"<)
    # database.appendData("Hello WOrld")