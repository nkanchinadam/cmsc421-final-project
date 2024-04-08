import tensorflow as tf

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

# MobileNetV2

IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

print(base_model)