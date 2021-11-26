import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


x_train = np.load(os.path.join('datasets','x_train.npy'))
x_test = np.load(os.path.join('datasets','x_test.npy'))
x_validation = np.load(os.path.join('datasets','x_validation.npy'))
y_train = np.load(os.path.join('datasets','y_train.npy'))
y_test = np.load(os.path.join('datasets','y_test.npy'))
y_validation = np.load(os.path.join('datasets','y_validation.npy'))

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")
