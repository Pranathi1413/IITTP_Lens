from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn
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
test_y = np.load("/content/gdrive/MyDrive/iittp_lens_images/datasets/y_test.npy")
test_x = np.load("/content/gdrive/MyDrive/iittp_lens_images/datasets/x_test.npy")
nsamples, nx, ny, nz = x_train.shape
d2_train_dataset = x_train.reshape((nsamples,nx*ny*nz))

nsamples, nx, ny, nz = test_x.shape
d2_test_dataset = test_x.reshape((nsamples,nx*ny*nz))

rf = RandomForestClassifier()
rf.fit(d2_train_dataset,y_train)


# Predictions on training and validation
y_pred_train = rf.predict(d2_train_dataset)
    # predictions for test
y_pred_test = rf.predict(d2_test_dataset)
    # training metrics
print("Training metrics:")
print(sklearn.metrics.classification_report(y_true= y_train, y_pred= y_pred_train))
    
    # test data metrics
print("Test data metrics:")
print(sklearn.metrics.classification_report(y_true=test_y, y_pred= y_pred_test))



  
  
  
