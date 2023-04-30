# Audify - The Audio Classifier

## Table of Contents

## Dataset Link
https://www.kaggle.com/datasets/chrisfilo/urbansound8k 

The dataset consist of 8732 audio files  in WAV format, 
the dataset includes 10 low level classes and number of files respectively

![WhatsApp Image 2023-04-30 at 20 06 53](https://user-images.githubusercontent.com/122106180/235359054-ca80aca6-f922-4663-b41a-9d5387528132.jpg)


## Tech Stack
### Programming Language: Python
### Python Libraries used
- Numpy
- Pandas
- Scikit
- Keras
- Tersorflow
- Librosa
- Flask
- Pickle
## Project Description
### Feature Extraction
In this project the feature considered is MFCC (Mel Frequency Cepstral Coefficients), MFCCs are a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear Mel frequency scale. The Mel frequency scale is a perceptual scale of frequency that is based on the human ear's response to sound. The power spectrum of the sound is divided into a number of frequency bands that are equally spaced on the Mel scale, and the energy in each band is then summed and logarithmically compressed. This log-compressed spectrum is then transformed using a Discrete Cosine Transform (DCT) to produce the MFCCs. MFCC is one of the commonly used features that has been
used in a variety of applications especially in voice signal processing such as speaker recognition, voice recognition, and
gender identification

The following code is used to extract MFCC value from each file

![WhatsApp Image 2023-04-30 at 20 14 26](https://user-images.githubusercontent.com/122106180/235359508-ce964475-7ebc-4a31-9332-298ebb2b110e.jpg)

the extracted MFCC values are made into Data Frame

![WhatsApp Image 2023-04-30 at 20 16 36](https://user-images.githubusercontent.com/122106180/235359774-00d53292-e0be-4c61-9e01-c48bbdae0323.jpg)

### Data Splitting
The created data set is split into 70-30 for Training and Testing of the Model

![WhatsApp Image 2023-04-30 at 20 33 29](https://user-images.githubusercontent.com/122106180/235360381-23ae2919-ea6a-4a88-aade-fb7452ac9052.jpg)

The Shape of the Training and Testing dataset

![WhatsApp Image 2023-04-30 at 20 34 15](https://user-images.githubusercontent.com/122106180/235360619-4db5662f-2e2f-487d-96fe-5969641066bb.jpg)

### Deep Neural Network Model

In the project we built fully connected network with an input layer ,2 hidden layers and an output layer using sequential API.
The first layer consists of 100 neurons and takes input of 40 features with dropout of 50% and uses RELU activation function.
The second layer  consists of 200 neurons  with dropout of 50% and uses RELU activation function to extract more complex features than the first layer.
The third layer is a dense layer with 100 neurons and a ReLU activation function, again extracting more complex features.
the last layer is another dense layer with the number of output classes (10) neurons and a softmax activation function.

![WhatsApp Image 2023-04-30 at 20 47 56](https://user-images.githubusercontent.com/122106180/235361081-3d88c2db-1816-4044-a035-46c8ccf61edd.jpg)

## What the project does?

## Images and Screenshots of the project
