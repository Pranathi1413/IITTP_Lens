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

# converting into y test and y predicted one hot vector
y_test = []
test_y = test_y.tolist()
for i in test_y:
  y_test.append(int(i[0]))
  
one_hot = np.zeros((270,27))
rows = np.arange(len(y_test))
one_hot[rows, y_test] = 1


y_pred = [int(a) for a in y_pred_test]
one_hot2 = np.zeros((270,27))
rows2 = np.arange(len(y_pred_test))
one_hot2[rows2, y_pred] = 1


# get current size
fig_size = plt.rcParams["figure.figsize"]

print ("Current size:", fig_size)
# let's make the plots a bit bigger than the default
# set figure width to 14 and height to 6
fig_size[0] = 14
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
print ("Current size:", fig_size)

# POC curver
fig, ax = plt.subplots()
precision = dict()
recall = dict()
for i in range(27):
    precision[i], recall[i], _ = precision_recall_curve(one_hot[:, i], one_hot2[:, i])
    ax.plot(recall[i], precision[i], lw=2)
    #ax.annotate(xy=(recall[i][-1][-1],precision[i][-1]), xytext=(27,0), text='class {}'.format(i))
    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()
# ROC curve
TPR = dict()
FPR = dict()
auc_classwise = []
for i in range(27):
    FPR[i], TPR[i], _ = roc_curve(one_hot[:, i], one_hot2[:, i])
    plt.plot(FPR[i], TPR[i], lw=2)
    auc_classwise.append(auc(FPR[i], TPR[i]))
    #ax.annotate(xy=(recall[i][-1][-1],precision[i][-1]), xytext=(27,0), text='class {}'.format(i))

    
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.legend(loc="best")
plt.title("ROC curve")
plt.show()
print(auc_classwise)


  
  
  
