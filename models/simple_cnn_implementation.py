"""Simple CNN implementation.ipynb

Original file is located at
    https://colab.research.google.com/drive/1A2EMLaE9UV9TPcuG9AQeozXikd01gfOj
"""

import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip','r')
zip_ref.extractall('/content')
zip_ref.close()

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout

train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/train',
    labels='inferred',
    batch_size = 32,
    image_size=(256,256)
)
validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/test',
    labels='inferred',
    batch_size = 32,
    image_size=(256,256)
)

#data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training Data Augmentation with Rescaling
train_datagen = ImageDataGenerator(
    rescale=1.0/255,           # Normalize pixel values to [0, 1]
    rotation_range=20,         # Random rotations
    width_shift_range=0.2,     # Horizontal shifts
    height_shift_range=0.2,    # Vertical shifts
    shear_range=0.2,           # Shearing transformation
    zoom_range=0.2,            # Zoom in/out
    horizontal_flip=True,      # Random horizontal flip
    fill_mode='nearest'        # Fill missing pixels with nearest neighbors
)

# Validation Data Preparation with Rescaling
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Load Training Data
train_ds = train_datagen.flow_from_directory(
    directory='/content/train',   # Path to training dataset
    target_size=(256, 256),       # Resize images to 256x256
    batch_size=32,                # Batch size
    class_mode='binary'           # Binary classification (cat vs dog)
)

# Load Validation Data
validation_ds = validation_datagen.flow_from_directory(
    directory='/content/test',    # Path to validation dataset
    target_size=(256, 256),       # Resize images to 256x256
    batch_size=32,                # Batch size
    class_mode='binary'           # Binary classification
)

def process(image,label):
  image = tf.cast(image/255,tf.float32)
  return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(BatchNormalization());
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization());
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization());
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_ds,epochs=10,validation_data=validation_ds)

import cv2
test_img = cv2.imread('/content/cat.jpg')
test_img = cv2.resize(test_img,(256,256))
test_input = test_img.reshape((1,256,256,3))
model.predict(test_input)