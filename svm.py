import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

clusters = []

x_train = np.load(os.path.join('datasets','x_train.npy'))
x_test = np.load(os.path.join('datasets','x_test.npy'))
x_validation = np.load(os.path.join('datasets','x_validation.npy'))
y_train = np.load(os.path.join('datasets','y_train.npy'))
y_test = np.load(os.path.join('datasets','y_test.npy'))
y_validation = np.load(os.path.join('datasets','y_validation.npy'))


param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)
model.fit(x_train,y_train)
# model.best_params_ contains the best parameters obtained from GridSearchCV

y_pred = model.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

'''
url=input('Enter URL of Image :')
img=imread(url)
plt.imshow(img)
plt.show()
img_resize=resize(img,(150,150,3))
l=[img_resize.flatten()]
probability=model.predict_proba(l)
for ind, val in enumerate(clusters):
    print(f'{val} = {probability[0][ind]*100}%')
print("The predicted image is : "+clusters[model.predict(l)[0]])

'''