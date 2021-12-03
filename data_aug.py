
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import skimage.io
import random
import os

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def vertical_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[ : , ::-1]

# dictionary of the transformations functions we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'vertical_flip': vertical_flip
}

folder_path = 'images'
# the number of file to generate
num_files_desired = 300

# loop on all files of the folder and build a list of files paths
images = os.listdir(folder_path)

num_generated_files = 1
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path =  random.choice(images)
    label = image_path.split('_')[0]
    image_path = folder_path + '/' + image_path
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)

    num_transformations_to_apply = random.randint(1, len(available_transformations))
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        # choose a random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)
        num_transformations += 1
    new_file_path = '%s/%s_augmented_image_%s.jpg' % (folder_path, label, num_generated_files)
    # write image to the disk
    sk.io.imsave(new_file_path, transformed_image)
    num_generated_files += 1