import pandas as pd 
import numpy as np
import IPython.display as ipd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#These are modules for audio processing
import librosa
import librosa.display  #audio visualization

audio_file_path="Urban Sound/fold1/101415-3-0-3.wav"
audio_data,sample_rate = librosa.load(audio_file_path)
     
print(audio_data)
plt.figure(figsize=(12,4))
plt.plot(audio_data)
mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
print(mfccs.shape)
import pandas as pd
import os

audio_dataset_path='Urban Sound'
metadata=pd.read_csv('Urban Sound/UrbanSound8K.csv')
metadata.head()
label=metadata['class']
metadata['class'].value_counts()
def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features
import numpy as np
from tqdm import tqdm

# Now we iterate through every audio file and extract features 
# using Mel-Frequency Cepstral Coefficients

extracted_features=[]

for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())
     
X.shape, y.shape
y=np.array(pd.get_dummies(y))
y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
def data_summary(X_train, y_train, X_test, y_test):
    """Summarize current state of dataset"""
    print('Training features :', X_train.shape)
    print('Training labels:', y_train.shape)
    print('Testing features:', X_test.shape)
    print('Test labels:', y_test.shape)
data_summary(X_train,y_train,X_test,y_test)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import Adam
from sklearn import metrics
num_labels=y.shape[1]
num_labels
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],optimizer='adam')
from keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 100
num_batch_size = 50

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)
test_accuracy=model.evaluate(X_test,y_test,verbose=0)
train_accuracy=model.evaluate(X_train,y_train,verbose=0)
print("Training accuracy: {:.2f}%".format(train_accuracy[1] * 100))
print("Test accuracy: {:.2f}%".format(test_accuracy[1] * 100))
X_test[1]
print("Classes are -\n",metadata['class'].unique())

from sklearn.preprocessing import LabelEncoder
filename="Urban Sound/fold6/194321-9-0-126.wav"
audio, sample_rate = librosa.load(filename) 
mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
le = LabelEncoder()
le.fit(label)
mfccs_mean = np.mean(mfccs.T,axis=0).reshape(1,-1)
#print(model.input_shape)
#print(mfccs_mean)
mfccs_scaled_features = mfccs_mean

#print(mfccs_scaled_features)
#print(mfccs_scaled_features.shape)
predicted_label=model.predict(mfccs_scaled_features)
print(predicted_label)
classes_x=np.argmax(predicted_label,axis=1)
prediction_class = le.inverse_transform(classes_x)
print("The predicted class label is ",prediction_class)
