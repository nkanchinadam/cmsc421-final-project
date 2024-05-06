import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import pandas as pd
import json
import numpy as np
from PIL import Image
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

IMG_SIZE_V2 = [200, 200]
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
      image = np.asarray(tf.image.resize(Image.open('./images/' + id + '.png').convert('RGB'), size=IMG_SIZE_V2, preserve_aspect_ratio=True)) / 255.0
      #image = np.asarray(Image.open('./images/' + id + '.png').convert('RGB').resize(IMG_SIZE)) / 256.0
    except:
      continue
    X.append(image)
    y.append(genre_data[id])
    count+=1
    if count == 8000:
      break

  X = np.asarray(X)
  y = np.asarray(y)
  print(X)

  y_labels = []
  for i in range(len(y)):
    y_labels.append(np.argmax(y[i],axis=0))

  y_labels = np.asarray(y_labels)

  X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=1, stratify=y_labels)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, stratify=y_train)


  # Revert back to original test data
  new_y_train = np.array([[0,0,0,0,0]])
  new_y_test = np.array([[0,0,0,0,0]])
  new_y_val = np.array([[0,0,0,0,0]])
  #y_train_count = [0,0,0,0,0]
  #y_test_count = [0,0,0,0,0]
  #y_val_count = [0,0,0,0,0]
  for data in y_train:
    if data == 0:
      new_y_train = np.concatenate((new_y_train, np.array([[1,0,0,0,0]])), axis=0)
      #y_train_count[0] += 1
    elif data == 1:
      new_y_train = np.concatenate((new_y_train, np.array([[0,1,0,0,0]])), axis=0)
      #y_train_count[1] += 1
    elif data == 2:
      new_y_train = np.concatenate((new_y_train, np.array([[0,0,1,0,0]])), axis=0)
      #y_train_count[2] += 1
    elif data == 3:
      new_y_train = np.concatenate((new_y_train, np.array([[0,0,0,1,0]])), axis=0)
      #y_train_count[3] += 1
    else:
      new_y_train = np.concatenate((new_y_train, np.array([[0,0,0,0,1]])), axis=0)
      #y_train_count[4] += 1

  for data in y_test:
    if data == 0:
      new_y_test = np.concatenate((new_y_test, np.array([[1,0,0,0,0]])), axis=0)
      #y_test_count[0] += 1
    elif data == 1:
      new_y_test = np.concatenate((new_y_test, np.array([[0,1,0,0,0]])), axis=0)
      #y_test_count[1] += 1
    elif data == 2:
      new_y_test = np.concatenate((new_y_test, np.array([[0,0,1,0,0]])), axis=0)
      #y_test_count[2] += 1
    elif data == 3:
      new_y_test = np.concatenate((new_y_test, np.array([[0,0,0,1,0]])), axis=0)
      #y_test_count[3] += 1
    else:
      new_y_test = np.concatenate((new_y_test, np.array([[0,0,0,0,1]])), axis=0)
      #y_test_count[4] += 1

  for data in y_val:
    if data == 0:
      new_y_val = np.concatenate((new_y_val, np.array([[1,0,0,0,0]])), axis=0)
      #y_val_count[0] += 1
    elif data == 1:
      new_y_val = np.concatenate((new_y_val, np.array([[0,1,0,0,0]])), axis=0)
      #y_val_count[1] += 1
    elif data == 2:
      new_y_val = np.concatenate((new_y_val, np.array([[0,0,1,0,0]])), axis=0)
      #y_val_count[2] += 1
    elif data == 3:
      new_y_val = np.concatenate((new_y_val, np.array([[0,0,0,1,0]])), axis=0)
      #y_val_count[3] += 1
    else:
      new_y_val = np.concatenate((new_y_val, np.array([[0,0,0,0,1]])), axis=0)
      #y_val_count[4] += 1

  new_y_train = new_y_train[1:]
  new_y_val = new_y_val[1:]
  new_y_test = new_y_test[1:]

  #print(y_train_count)
  #print(y_test_count)
  #print(y_val_count)

  datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    #shear_range=0.1,
    #zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    #fill_mode='nearest'
  )

  datagen.fit(X_train)
  
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
  
  X_train = np.asarray(X_train)
  model.fit(datagen.flow(X_train, new_y_train, batch_size=BATCH_SIZE), validation_data=(X_val, new_y_val), epochs=5, verbose=2, callbacks=[early_stopping])
  
  
  test_pred = model.predict(X_test)
  
  test_pred_label = []
  y_test_label = []
  for i in range(len(test_pred)):
    test_pred_label.append(np.argmax(test_pred[i], axis=0))
    y_test_label.append(np.argmax(new_y_test[i], axis=0))

  report = classification_report(y_test_label, test_pred_label)
  
  print(report)
  
  model.save("5GenreModel.keras")
  
if __name__ == "__main__":
  main()