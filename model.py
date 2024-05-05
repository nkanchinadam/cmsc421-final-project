import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import pandas as pd
import json
import numpy as np
from PIL import Image
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import keras.src.legacy.backend as K
import keras.backend as K2

IMG_SIZE = (200, 200)
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 32

def get_f1(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.cast(y_pred, tf.float32)
  true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  predicted_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_pos / (predicted_positive + K2.epsilon())
  recall = true_pos / (possible_positives + K2.epsilon())
  f1 = 2 * (precision * recall) / (precision + recall + K2.epsilon())
  return f1

def main():
  X = []
  y = []
  count = 0
  genre_data = json.load(open('./new_data/5LabelGenreDataDistinct.json', 'r'))
  for id in genre_data.keys():
    image = None
    try:
      image = np.asarray(Image.open('./images/' + id + '.png').convert('RGB').resize(IMG_SIZE)) / 256.0
    except:
      continue
    X.append(image)
    y.append(genre_data[id])
    count+=1
    if count == 8000:
      break

  X = np.array(X)
  y = np.array(y)
  
  y_labels = []
  for i in range(len(y)):
    y_labels.append(np.argmax(y[i],axis=0))

  y_labels = np.array(y_labels)
  # print(X)
  print(y_labels)

  X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=1, stratify=y_labels)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, stratify=y_train)

  # print(X_test)
  print(y_test)
  # print(X_val)
  print(y_val)

  #datagen = ImageDataGenerator(
    #rotation_range=10,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #shear_range=0.1,
    #zoom_range=0.1,
    #horizontal_flip=True,
    #vertical_flip=True,
    #fill_mode='nearest'
  #)

  #datagen.fit(X_train)
  
  base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

  for layer in base_model.layers:
    layer.trainable = False

  model = Sequential()
  model.add(base_model)
  model.add(Flatten())
  
  #model.add(Dropout(0.1))
  model.add(Dense(1024, activation='relu'))#, kernel_regularizer=l2(0.001)))
  #model.add(Dropout(0.1))
  #model.add(Dense(512, activation='relu'))#, kernel_regularizer=l2(0.001)))
  #model.add(Dropout(0.1))
  #model.add(Dense(256, activation='relu'))#, kernel_regularizer=l2(0.1)))
  #model.add(Dropout(0.1))
  #model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.1)))
  #model.add(Dropout(0.1))
  #model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.1)))
  model.add(BatchNormalization())
  
  model.add(Dense(5, activation='softmax'))
  
  # optimizer can be fine tuned
  optimizer = Adam(learning_rate=0.01)
  
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[get_f1])
  print(model.summary())
  
  # Early stopping
  early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
  
  model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), epochs=5, verbose=2, callbacks=[early_stopping])
  
  
  test_pred = model.predict(X_test)
  
  test_pred_label = []
  y_test_label = []
  for i in range(len(test_pred)):
    test_pred_label.append(np.argmax(test_pred[i], axis=0))
    y_test_label.append(np.argmax(y_test[i], axis=0))

  report = classification_report(y_test_label, test_pred_label)
  
  print(report)
  
  model.save("5GenreModel.keras")
  
if __name__ == "__main__":
  main()