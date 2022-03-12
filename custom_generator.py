import imp
import os
from tkinter import Image
import pandas as pd
import xml.etree.ElementTree as ET
import glob
import numpy as np
import tensorflow as tf

np.set_printoptions(suppress=True)

def xml_to_csv(path):

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (os.path.join(path, filename),
                     width,
                     height,
                     member.find('name').text,
                     int(bndbox.find('xmin').text),
                     int(bndbox.find('ymin').text),
                     int(bndbox.find('xmax').text),
                     int(bndbox.find('ymax').text),
                    )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    return xml_df

#xml to array
#impart array in train si test

def xml_to_array(path):
    X_array = []
    Y_array = []

    for image in glob.glob(path + '/*.jpg'):
        xml_file = image.replace(".jpg", ".xml")
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            label = member.find('name').text
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            img = tf.keras.preprocessing.image.load_img(image)
            img_arr = tf.keras.preprocessing.image.img_to_array(img)

            img_arr = img_arr[ymin:ymax, xmin:xmax]
            img_arr = tf.image.resize(img_arr,(150, 150)).numpy()

            X_array.append(img_arr)
            label = 0 if label=='available' else 1
            Y_array.append(label)
    
    X_array = np.asarray(X_array, dtype='float32')
    Y_array = np.asarray(Y_array)

    return X_array, Y_array




class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col, batch_size, input_size=(150, 150, 3), shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.n = len(self.df)
        self.n_class = df[y_col['name']].nunique()
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, path, bbox, target_size):
    
        xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = image_arr[ymin:ymax, xmin:xmax]
        image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()
        #image_arr = np.array(image_arr)

        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        # print(f'label is {label}')
        return label #tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col['path']]
        bbox_batch = batches[self.X_col['bbox']]
        
        name_batch = batches[self.y_col['name']]

        X_batch = np.asarray([self.__get_input(x, y, self.input_size) for x, y in zip(path_batch, bbox_batch)])

        y0_batch = np.asarray([self.__get_output(y, self.n_class) for y in name_batch])
        
        return X_batch, y0_batch
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size