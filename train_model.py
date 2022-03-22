# %%
import os
from black import Mode
from keras import layers, Model
# %%
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
from keras.applications.inception_v3 import InceptionV3

local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3), include_top=False, weights=None)
pre_trained_model.load_weights(local_weights_file)
# %%

for layer in pre_trained_model.layers:
  layer.trainable = False
# %%

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output
# %%

from tensorflow.keras.optimizers import RMSprop
import keras

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

# Configure and compile the model
model = Model(pre_trained_model.input, x)

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.0001),
              metrics=['accuracy'],
              run_eagerly=True)

#model.summary()
# %%

from keras.preprocessing.image import ImageDataGenerator
from custom_generator import *

base_dir = 'imagini'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

batch_size = 4

datagen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1)

X_train, Y_train = xml_to_array(train_dir)

# %%

model.fit(datagen.flow(X_train, Y_train, batch_size=32,
         subset='training'),
         validation_data=datagen.flow(X_train, Y_train,
         batch_size=4, subset='validation'),
         steps_per_epoch=len(X_train) // 32, 
         epochs=15,
         shuffle=True)


# %%
#predictie
import matplotlib.pyplot as plt

x_test, y_test = xml_to_array(validation_dir)

image = x_test[4]/255.
print("label era {}".format(y_test[4]))
plt.imshow(image)
plt.show()
image = np.expand_dims(image, axis=0)
model.predict(image, verbose=1, batch_size = 1)

# %%
#salvare model
model.save("model_v1_1.h5")
# %%
#incarcare model
import keras

trained_model = Model()
trained_model = keras.models.load_model("model_v1_1.h5")

# %%
import matplotlib.pyplot as plt
from custom_generator import xml_to_array
import numpy as np

base_dir = 'imagini'
validation_dir = os.path.join(base_dir, 'validation')

x_test, y_test = xml_to_array(validation_dir)

image = x_test[3]/255.
print("label era {}".format(y_test[3]))
plt.imshow(image)
plt.show()
image = np.expand_dims(image, axis=0)
trained_model.predict(image, verbose=1, batch_size = 1)
# %%
