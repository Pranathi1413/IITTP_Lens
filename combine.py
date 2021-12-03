from tqdm import tqdm
import os
import shutil

#just run this if you wanna have all the images in one folder with cluster index at front

no_of_classes = 27

def combine_clusters(direc):
    features = []
    img_name = []
    #i is cluster folder name
    for i in tqdm(direc):
        imges = os.listdir('clusters/' + i)
        label = int(i[1:]) - 1
        label = str(label)
        #j is image name
        for j in tqdm(imges):
            shutil.copyfile(os.path.join('clusters/' + i , j ), 'images/' + label + "_" + j )


cluster_path = os.listdir('clusters')
combine_clusters(cluster_path)

# classname = [0:'TC-1',  1:'Classroom Complex',  2:'CS Lab', 3:'Cricket/Football ground', 4:'Front', 5:'Girls hostel', 6:'Guest house', 7:'Gym', 8:'Health center', 9:'Hostel', 10:'Indoor Stadium', 11:'Lab', 12:'Library', 13:'Mess', 14:'OAT steps', 15:'Outdoor courts', 16:'Parking lot', 17:'Roads', 18:'Classroom Complex', 19:'Classroom Complex', 20:'Hostels', 21:'Hostel', 22:'Indoor Stadium', 23:'Lab', 24:'Library', 25:'OAT', 26:'TC-22']