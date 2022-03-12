# %%
from itertools import count
import os

from keras import layers
from keras import Model
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
x = layers.Dense(256, activation='relu')(x)
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

# Define our example directories and files
base_dir = 'imagini'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

"""
# train_df = xml_to_csv(train_dir)
# val_df = xml_to_csv(validation_dir)

# col=[]
# for index, row in train_df.iterrows():
#     val = {'xmin': train_df.at[index,'xmin'],
#             'ymin': train_df.at[index,'ymin'],
#             'xmax': train_df.at[index,'xmax'],
#             'ymax': train_df.at[index,'ymax']}

#     col.append(val)

# train_df["coordinates"] = col
# train_df['class'] = pd.factorize(train_df['class'])[0]

# col_1 = []
# for index, row in val_df.iterrows():
#     val = {'xmin':val_df.at[index,'xmin'],
#             'ymin':val_df.at[index,'ymin'],
#             'xmax':val_df.at[index,'xmax'],
#             'ymax':val_df.at[index,'ymax']}
#     col_1.append(val)

# val_df["coordinates"] = col_1
# val_df['class'] = pd.factorize(val_df['class'])[0]
"""

batch_size = 4

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1)


X_train, Y_train = xml_to_array(train_dir)

# print(X_train.shape)
# print(X_train[0].shape)
# print(Y_train.shape)
# print(Y_train[0])

#os.chdir("C:\\Users\\Alex\\Desktop\\LICENTa\\ML MODEL")


# %%

model.fit(datagen.flow(X_train, Y_train, batch_size=32,
         subset='training'),
         validation_data=datagen.flow(X_train, Y_train,
         batch_size=4, subset='validation'),
         steps_per_epoch=len(X_train) // 32, 
         epochs=15)


# %%
#generator
import matplotlib.pyplot as plt
nrows = 3
ncols = 3
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

for i in range(0, 4):
    sp = plt.subplot(nrows, ncols, i+ 1)
    item = training_data.__getitem__(i)
    var = item[0][0]
    plt.imshow(var)
    #print(item[1][0][0])

# %%
#prediction
import matplotlib.pyplot as plt

var = validation_generator.__getitem__(0)
var = var[0][0]
print(var.shape)
plt.imshow(var)
var = np.expand_dims(var, axis=0)
model.predict(var, verbose=1, batch_size = 1)
# %%
import matplotlib.pyplot as plt

x_test, y_test = xml_to_array(validation_dir)
#print(x_test[0].shape)

image = x_test[0]/255.
print("label era {}".format(y_test[0]))
plt.imshow(image)
plt.show()
image = np.expand_dims(image, axis=0)
model.predict(image, verbose=1, batch_size = 1)

# %%
