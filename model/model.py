import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.utils import np_utils 
from keras.initializers import RandomNormal
from keras.models import Sequential, model_from_json 
from keras.layers import Dense, Activation
from numpy import array,argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

cotton = pd.read_csv('leaf_dataset.csv')

name = cotton['name']
label = cotton.disease_detected
features = cotton.drop(['disease_detected','name'],axis=1)

values = array(label)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
label = onehot_encoded

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(8,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(features, label, epochs=100, batch_size=None)

scores = model.evaluate(features, label, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")