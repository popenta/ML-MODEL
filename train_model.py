# %%
import os

from tensorflow.keras import layers
from tensorflow.keras import Model
# %%
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
#experimental_run_tf_function=False
#tf.compat.v1.enable_eager_execution()
#tf.config.run_functions_eagerly(True)

from tensorflow.keras.applications.inception_v3 import InceptionV3

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
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

# Configure and compile the model
model = Model(pre_trained_model.input, x)

# model = keras.models.Sequential([
#   keras.layers.Flatten(),
#   keras.layers.Dense(128, activation='relu'),
#   keras.layers.Dropout(0.2),
#   keras.layers.Dense(2, activation='sigmoid')
# ])
# model.add(keras.layers.Flatten())

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.0001),
              metrics=['acc'],
              run_eagerly=True)

#model.summary()
# %%

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from custom_generator import *

# Define our example directories and files
base_dir = 'imagini'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_df = xml_to_csv(train_dir)
val_df = xml_to_csv(validation_dir)

col=[]
for index, row in train_df.iterrows():
    val = {'xmin': train_df.at[index,'xmin'],
            'ymin': train_df.at[index,'ymin'],
            'xmax': train_df.at[index,'xmax'],
            'ymax': train_df.at[index,'ymax']}

    col.append(val)

train_df["coordinates"] = col
train_df['class'] = pd.factorize(train_df['class'])[0]
#train_df['class'] = np.asarray(train_df['class']).astype('float32').reshape((-1,1))

col_1 = []
for index, row in val_df.iterrows():
    val = {'xmin':val_df.at[index,'xmin'],
            'ymin':val_df.at[index,'ymin'],
            'xmax':val_df.at[index,'xmax'],
            'ymax':val_df.at[index,'ymax']}
    col_1.append(val)

val_df["coordinates"] = col_1
val_df['class'] = pd.factorize(val_df['class'])[0]
#val_df['class'] = np.asarray(val_df['class']).astype('float32').reshape((-1,1))

batch_size = 1

os.chdir(train_dir)
train_generator = CustomDataGen(train_df,
                                X_col = {'path': 'filename', 'bbox' : 'coordinates'},
                                y_col = {'name': 'class'},
                                batch_size = batch_size)



os.chdir('C:\\Users\\Alex\\Desktop\\LICENTa\\ML MODEL\\imagini\\validation\\')
validation_generator = CustomDataGen(val_df,
                                X_col = {'path': 'filename', 'bbox' : 'coordinates'},
                                y_col = {'name': 'class'},
                                batch_size = batch_size)

os.chdir("C:\\Users\\Alex\\Desktop\\LICENTa\\ML MODEL")

"""
# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 300),
        batch_size=20,
        class_mode='binary')
"""

# %%

history = model.fit(
      train_generator,
      steps_per_epoch=100,
      epochs=2,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)
# %%
import matplotlib.pyplot as plt
# nrows = 3
# ncols = 3
# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, nrows * 4)

# for i in range(0, 4):
#     sp = plt.subplot(nrows, ncols, i+ 1)
#     item = train_generator.__getitem__(i)
#     var = item[0][0]
#     plt.imshow(var)
#     print(item[1][0][0])

# %%
#prediction

var = validation_generator.__getitem__(12)
var = var[0][0]
plt.imshow(var)
var = np.expand_dims(var, axis=0)
model.predict(var, verbose=1, batch_size = 1)
# %%
