FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN apt-get install -y vim
RUN apt-get install -y libsndfile1
RUN python3 -m pip install torch==1.5.0
RUN git clone https://github.com/SuyashMore/MevonAI-Speech-Emotion-Recognition.git
WORKDIR "/MevonAI-Speech-Emotion-Recognition/src"
RUN chmod +x setup.sh
RUN ./setup.sh
RUN apt-get install -y python3-pyaudio
