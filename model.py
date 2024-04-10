import tensorflow as tf
import pandas as pd
import json

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

def main():
  
  df = pd.read_csv('./movies/MovieGenre.csv', encoding='ISO-8859-1')

  with open("pixelData.json", "r") as file:
    pixelData = json.load(file)
  print(pixelData)
  # MobileNetV2

  IMG_SHAPE = IMG_SIZE + (3,)
  base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
  print(base_model)
  
if __name__ == "__main__":
  main()