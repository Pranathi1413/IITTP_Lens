from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread

def createImageFeatures(image, size=(244, 244, 3)):
    # resize the image
    image = resize(image, size)
    # flatten the image
    pixel_list = image.flatten()
    return pixel_list

n_classes = 19
clusters = []
'''
flat_data_arr=[] #input array
target_arr=[] #output array
datadir='/content/drive/MyDrive/ML' 
#path which contains all the categories of images
for i in range(n_classes):
    print(f'loading... category : {i}')
    fname = "cluster" + str(i)
    path=os.path.join(datadir,fname)
    for img in os.listdir(path):
        img_array=imread(os.path.join(path,img))
        img_resized=resize(img_array,(244,244,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(i)
    print(f'loaded category:{i} successfully')
'''
raw_images = np.load(os.path.join("mydatasets","X"))
labels = np.load(os.path.join("mydatasets","y"))

(train_X, test_X, train_y, test_y) = train_test_split(raw_images, labels, test_size=0.25, random_state=0)
    
model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_X, train_y)
acc = model.score(test_X, test_y)
print("Raw pixel accuracy: {:.2f}%".format(acc * 100))

'''
image = cv2.imread("dog.jpg")
dog = createImageFeatures(image)
dog = np.array([dog])
print(model.predict(dog))
'''