import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

n_classes = 27
clusters = []
'''
flat_data_arr=[] #input array
target_arr=[] #output array
datadir='clusters' 
#path which contains all the categories of images
for i in range(n_classes):
    j = 0
    print(f'loading... category : {i+1}')
    fname = "c" + str(i+1)
    path=os.path.join(datadir,fname)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(244,244,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(i+1)
    print(f'loaded category:{i+1} successfully')

np.save(os.path.join("mydatasets","X"), flat_data)
np.save(os.path.join("mydatasets","Y"), target)
'''

flat_data=np.load(os.path.join("mydatasets","X.npy"))
target=np.load(os.path.join("mydatasets","Y.npy"))

print("data loaded")

df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
x=df.iloc[:,:-1] #input data 
y=df.iloc[:,-1] #output data

param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=77,stratify=y)
print('Splitted Successfully')
model.fit(x_train,y_train)
print('The Model is trained well with the given images')
# model.best_params_ contains the best parameters obtained from GridSearchCV

print("params:")
print(model.best_params_)
y_pred=model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

'''
url=input('Enter URL of Image :')
img=imread(url)
plt.imshow(img)
plt.show()
img_resize=resize(img,(150,150,3))
l=[img_resize.flatten()]
probability=model.predict_proba(l)
for ind,val in enumerate(clusters):
    print(f'{val} = {probability[0][ind]*100}%')
print("The predicted image is : "+clusters[model.predict(l)[0]])
'''