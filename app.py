from flask import Flask, request, render_template
import librosa
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
app = Flask(__name__)
model = pickle.load(open('D:/Audify/7.1 Audio Classify Code and Files/Code and Files/models/model.pkl', 'rb'))

@app.route('/')
def home():
   classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
   return render_template('index.html', classes=classes)

@app.route('/classify-audio', methods=['POST'])
def classify_audio():
    # Load the audio file from the HTML form
    #audio_file = request.files['audio-file']
    print(request.files)
    audio_file = request.files.get('audio-file')
    #audio_file = request.files['audio-file']
    print(audio_file)
    # Preprocess the audio data to extract the Mel spectrogram coefficients
    y, sr = librosa.load(audio_file)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000)
    log_S = librosa.power_to_db(S, ref=np.max)
    log_S = log_S.reshape(1, -1)
    log_S = log_S[:, :40]
    
    # Use the model to make a prediction
    prediction = model.predict(log_S)
    
    # Map the prediction index to the corresponding class label
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    max_prob_index = np.argmax(prediction)
    predicted_class = classes[max_prob_index]
    
    return render_template('results.html', predicted_class=predicted_class, num_classes=len(classes), classes=classes)

if __name__ == '__main__':
    app.run(debug=True)
