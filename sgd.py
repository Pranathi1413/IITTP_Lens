import os
from sklearn.linear_model import SGDClassifier
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_curve, accuracy_score, precision_recall_curve

n_classes = 27
x=np.load(os.path.join("datasets","x_train.npy"))
y=np.load(os.path.join("datasets","y_train.npy"))
x = np.concatenate((x,np.load(os.path.join("datasets","x_validation.npy"))))
y = np.concatenate((y,np.load(os.path.join("datasets","y_validation.npy"))))
nsamples, nx, ny, nz = x.shape
x = x.reshape((nsamples,nx*ny*nz))
y = preprocessing.label_binarize(y, classes=range(27))
x_train, y_train = x, y
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)

x=np.load(os.path.join("datasets","x_test.npy"))
y=np.load(os.path.join("datasets","y_test.npy"))
nsamples, nx, ny, nz = x.shape
x = x.reshape((nsamples,nx*ny*nz))
y = preprocessing.label_binarize(y, classes=range(27))
x_test, y_test = x, y
print("data loaded")

sgd_clf = OneVsRestClassifier( SGDClassifier( max_iter=100, tol=1e-3))
sgd_clf.fit(x_train, y_train)
y_pred = sgd_clf.predict(x_test)
print('accuracy: ', 100*accuracy_score(y_test, y_pred))

print ("here")
y_score = sgd_clf.decision_function(x_test)


print("dumping")
filename = 'sgd_model.sav'
pickle.dump(sgd_clf, open(filename, 'wb'))


fpr = dict()
tpr = dict()
#roc_auc = dict()
plt.figure()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    plt.plot(fpr[i], tpr[i], lw=1, label='class {}'.format(i))
    #roc_auc[i] = auc(fpr[i], tpr[i])
    print("hmm")
plt.xlabel("False positive")
plt.ylabel("True positive")
plt.legend(loc=7)
plt.title("ROC curve")
#plt.show()   
 
 
precision = dict()
recall = dict()
plt.figure()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    plt.plot(recall[i], precision[i], lw=1, label='class {}'.format(i))
    #, label='class {}'.format(i))
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc=7)
plt.title("precision vs. recall curve")
plt.show()
