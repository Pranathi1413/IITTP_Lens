from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.applications.vgg19 import preprocess_input
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
# import pandas as pd
import os
# import cv2
import matplotlib.pyplot as plt
import pickle
# train_path="/content/gdrive/MyDrive/iittp_lens_images/anotherDS"
x_train = np.load(os.path.join('datasets','x_train.npy'))
x_val = np.load(os.path.join('datasets',"x_validation.npy"))
y_train = np.load(os.path.join('datasets',"y_train.npy"))
y_val = np.load(os.path.join('datasets',"y_validation.npy"))
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
test_y = np.load(os.path.join('datasets',"y_test.npy"))
test_x = np.load(os.path.join('datasets',"x_test.npy"))
model.evaluate(x_val,y_val,batch_size=10)
filename = 'vgg_model.sav'
pickle.dump(model, open(filename, 'wb'))

y_score = model.predict(test_x)
y_score.shape
 # converting ytest to one hot vector
y_test = []
test_y = test_y.tolist()
for i in test_y:
  y_test.append(int(i[0]))
  
#print(y_test)
one_hot = np.zeros((270,27))
rows = np.arange(len(y_test))
one_hot[rows, y_test] = 1

# get current size
fig_size = plt.rcParams["figure.figsize"]

#plotting graphs
print ("Current size:", fig_size)
# let's make the plots a bit bigger than the default
# set figure width to 14 and height to 6
fig_size[0] = 14
fig_size[1] = 6
plt.rcParams["figure.figsize"] = fig_size
print ("Current size:", fig_size)

#Pr curve
fig, ax = plt.subplots()
precision = dict()
recall = dict()
for i in range(27):
    precision[i], recall[i], _ = precision_recall_curve(one_hot[:, i], y_score[:, i])
    ax.plot(recall[i], precision[i], lw=2)
    #ax.annotate(xy=(recall[i][-1][-1],precision[i][-1]), xytext=(27,0), text='class {}'.format(i))
    
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()

#ROC curve and score
TPR = dict()
FPR = dict()
auc_classwise = []
for i in range(27):
    FPR[i], TPR[i], _ = roc_curve(one_hot[:, i], y_score[:, i])
    plt.plot(FPR[i], TPR[i], lw=2)
    auc_classwise.append(auc(FPR[i], TPR[i]))
    #ax.annotate(xy=(recall[i][-1][-1],precision[i][-1]), xytext=(27,0), text='class {}'.format(i))

    
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.legend(loc="best")
plt.title("ROC curve")
plt.show()
print(auc_classwise)

