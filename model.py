import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import pandas as pd
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten

IMG_SIZE = (200, 200)
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 32

def main():
  X = []
  y = []
  shapes = set()
  for name in ['Nitin']:#, 'Michael', 'Jihyo', 'Jason', 'Cara']:
    genre_data = json.load(open('./data/genreData' + name + '.json', 'r'))
    for id in genre_data.keys():
      image = None
      try:
        image = np.asarray(Image.open('./images/' + id + '.png').convert('RGB').resize(IMG_SIZE)) / 256.0
        shapes.add(image.shape)
      except:
        continue

      X.append(image)
      y.append(genre_data[id])
  print(shapes)
  X = np.array(X)
  y = np.array(y)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


  base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

  model = Sequential()
  model.add(base_model)
  model.add(Flatten())
  model.add(Dense(28, activation='softmax'))
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  print(model.summary())
  model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=20, verbose=2)
  
if __name__ == "__main__":
  main()