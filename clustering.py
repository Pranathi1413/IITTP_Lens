from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil


def image_feature(direc):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = []
    img_name = []
    for i in tqdm(direc):
        fname = 'mlproj' + '/' + i
        img = image.load_img(fname, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x)
        feat = feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features, img_name


img_path = os.listdir('mlproj')
img_features, img_name = image_feature(img_path)
image_cluster = pd.DataFrame(img_name, columns=['image'])

# Creating Clusters
k = 16
clusters = KMeans(k, random_state=40)
clusters.fit(img_features)
# To mention which image belong to which cluster
image_cluster["clusterid"] = clusters.labels_

# Made folder to seperate images
for i in range(k):
    os.makedirs('clusters/cluster' + str(i))

# Images will be seperated according to cluster they belong
for i in range(len(image_cluster)):
    shutil.copyfile(os.path.join(
        'mlproj', image_cluster['image'][i]), 'clusters/cluster' + str(image_cluster['clusterid'][i]) + "/" + image_cluster['image'][i])
