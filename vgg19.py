from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import pickle
train_path="/content/gdrive/MyDrive/iittp_lens_images/anotherDS"
x_train = np.load("/content/gdrive/MyDrive/iittp_lens_images/anotherDS/x_train.npy")
x_val = np.load("/content/gdrive/MyDrive/iittp_lens_images/anotherDS/x_validation.npy")
y_train = np.load("/content/gdrive/MyDrive/iittp_lens_images/anotherDS/y_train.npy")
y_val = np.load("/content/gdrive/MyDrive/iittp_lens_images/anotherDS/y_validation.npy")
vgg = VGG19(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False
x = Flatten()(vgg.output)
#adding output layer.Softmax classifier is used as it is multi-class classification
prediction = Dense(27, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()
model.compile(
  loss='sparse_categorical_crossentropy',
  optimizer="adam",
  metrics=['accuracy']
)

early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
#Early stopping to avoid overfitting of model
# fit the model
history = model.fit(
  x_train,
  y_train,
  validation_data=(x_val,y_val),
  epochs=20,
  callbacks=[early_stop],
  batch_size=10,shuffle=True)
test_y = np.load("/content/gdrive/MyDrive/iittp_lens_images/datasets/y_test.npy")
test_x = np.load("/content/gdrive/MyDrive/iittp_lens_images/datasets/x_test.npy")
model.evaluate(x_val,y_val,batch_size=10)
filename = 'vgg_model.sav'
pickle.dump(model, open(filename, 'wb'))



