[![PR](https://camo.githubusercontent.com/f96261621753dacf526590825b84f87ccb1db0e6/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5052732d77656c636f6d652d627269676874677265656e2e7376673f7374796c653d666c6174)](pullreq-url)
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="#">
    <img src="https://i.imgur.com/f1TqviT.jpeg" alt="Logo">
  </a>

  <h3 align="center">MevonAI - Speech Emotion Recognition</h3>

  <p align="center">
    Identify the emotion of multiple speakers in a Audio Segment 
    <br />
    <br />
    <a href="https://colab.research.google.com/drive/1RG8Ms2M7GKro2GPJluy3UwpuOj4WCXgf?usp=sharing">Try the Demo</a>
    ·
    <a href="https://github.com/SuyashMore/MevonAI-Speech-Emotion-Recognition/issues">Report Bug</a>
    ·
    <a href="https://github.com/SuyashMore/MevonAI-Speech-Emotion-Recognition/issues">Request Feature</a>
  </p>
</p>


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Installation](#installation)
  * [Running the Application](#Running-the-Application)
* [How it Works](#Here's-how-it-works)
  * [Speaker Diarization](#Speaker-Diarization)
  * [Feature Extraction](#Feature-Extraction)
  * [CNN Model](#CNN-Model)
  * [Training the Model](#Training-the-Model)
* [Contributing](#Contributing)
* [License](#license)
* [Acknowledgements](#acknowledgements)
* [FAQ](#faq)





<!-- ABOUT THE PROJECT -->
## About The Project

<img src="https://i.imgur.com/xaY8Izs.png" alt="Logo">

The main aim of the project is to Identify the emotion of multiple speakers in a call audio as a application for customer satisfaction feedback in call centres.


### Built With

* [Python 3.6.9](https://www.python.org/downloads/release/python-369/) 
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
* [Tensorflow-Keras](https://www.tensorflow.org/guide/keras/functional)
* [librosa](https://github.com/librosa/librosa)


<!-- GETTING STARTED -->
## Getting Started
Follow the Below Instructions for setting the project up on your local Machine.


### Installation

1. Clone the repo
```sh
git clone https://github.com/your_username_/Project-Name.git
```
2. Install Dependencies
```sh
sudo chmod +x src/setup.sh
./setup.sh
```

### Running the Application

1. Add audio files in .wav format for analysis in src/input/ folder

2. Run Speech Emotion Recognition using 
```sh
python3 src/speechEmotionRecognition.py
```
3. By Default , the application will use the Pretrained Model Available in "src/model/"

4. Diarized files will be sored in "src/output/" folder

5. Predicted Emotions will be stored in a separate .csv file in src/ folder


## Here's how it works:

#### Speaker Diarization
* Speaker diarisation (or diarization) is the process of partitioning an input audio stream into homogeneous segments according to the speaker identity. It can enhance the readability of an automatic speech transcription  by structuring the audio stream into speaker turns and, when used together with speaker recognition systems, by providing the speaker’s true identity. It is used to answer the question "who spoke when?" Speaker diarisation is a combination of speaker segmentation and speaker clustering. The first aims at finding speaker change points in an audio stream. The second aims at grouping together speech segments on the basis of speaker characteristics.

<img src="https://github.com/taylorlu/Speaker-Diarization/raw/master/resources/diarization.gif" alt="Logo">


#### Feature Extraction
* When we do Speech Recognition tasks, MFCCs is the state-of-the-art feature since it was invented in the 1980s.This shape determines what sound comes out. If we can determine the shape accurately, this should give us an accurate representation of the phoneme being produced. The shape of the vocal tract manifests itself in the envelope of the short time power spectrum, and the job of MFCCs is to accurately represent this envelope. 

<img src="https://i.imgur.com/UANHXoU.png" alt="Logo">
The Above Image represents the audio Waveform , the below image shows the converted MFCC Output on which we will Run our CNN Model.


#### CNN Model
* Use Convolutional Neural Network to recognize emotion on the MFCCs with the following Architecture
```python
model_ravdess = Sequential()
kernel = 5
model_ravdess.add(Conv2D(32, 5,strides=2,padding='same',
                 input_shape=(13,216,1)))
model_ravdess.add(Activation('relu'))
model_ravdess.add(BatchNormalization())


model_ravdess.add(Conv2D(64, 5,strides=2,padding='same',))
model_ravdess.add(Activation('relu'))
model_ravdess.add(BatchNormalization())

model_ravdess.add(Conv2D(64, 5,strides=2,padding='same',))
model_ravdess.add(Activation('relu'))
model_ravdess.add(BatchNormalization())

model_ravdess.add(Flatten())


model_ravdess.add(Dense(7))
model_ravdess.add(Activation('softmax'))

opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
```





<!-- USAGE EXAMPLES -->
## Training the Model

* [Download RAVDESS Emotional speech audio dataset ](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

* [2DConvolution.ipynb](https://github.com/SuyashMore/MevonAI-Speech-Emotion-Recognition/blob/master/src/notebooks/2D_Convolution.ipynb) file is used to training the model

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Speech Emotion Recognition from Saaket Agashe's Github](https://github.com/saa1605/speech-emotion-recognition)
* [Speech Emotion Recognition with CNN](https://towardsdatascience.com/speech-emotion-recognition-with-convolution-neural-network-1e6bb7130ce3)
* [MFCCs Tutorial](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
* [UIS-RNN Fully Supervised Speaker Diarization](https://github.com/google/uis-rnn)
* [uis-rnn and speaker embedding by vgg-speaker-recognition by taylorlu](https://github.com/taylorlu/Speaker-Diarization)


## FAQ

- **How do I do *specifically* so and so?**
    - Create an Issue to this repo , we will respond to the query


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=flat-square
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=flat-square
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=flat-square
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
