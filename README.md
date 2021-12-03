# IITTP_Lens

* ML Model to identify (to the best of our model's accuracy) the hang-out spots at IIT Tirupati Campus. 
* It uses a VGG16, a CNN model, as of now, and is limited to predicting a class of 27 places.

* It is currently deployed at: https://iittp-lens.herokuapp.com/


## The Classes
```
0:'TC-1',  1:'Classroom Complex',  2:'CS Lab', 3:'Cricket/Football ground', 4:'Front', 5:'Girls hostel', 6:'Guest house', 7:'Gym', 8:'Health center', 9:'Hostel', 10:'Indoor Stadium', 11:'Lab', 12:'Library', 13:'Mess', 14:'OAT steps', 15:'Outdoor courts', 16:'Parking lot', 17:'Roads', 18:'Classroom Complex', 19:'Classroom Complex', 20:'Hostels', 21:'Hostel', 22:'Indoor Stadium', 23:'Lab', 24:'Library', 25:'OAT', 26:'TC-22'
```

### Vgg16 
> This is a Convolutional Neural Network (CNN) model, which has been implemented over AlexNet by replacing large kernel sized filters with multiple 3X3 kernel sized filters. Using transfer learning technique, we have used Vgg16 pre-trained model for image classification in 27 classes.
```sh
Training Accuracy : 100%
Test Acccuracy : 87.93%
Average classiwse ROC-AUC Score: 0.962
```  
Based on our research, we found that Vgg16 model perfromed better than any other exisiting model for image classification. Here, auc score is almost close to 1 means model able to correctly classify, majority of the images to their respective classes.

### MLP-mixer
>This is a Multi Layer Perceptron (MLP) model for computer vision that uses two types of layers. The image is split into square patches. One layer acts on each patch individually while the other layer is applied across all patches. 

The model was observed to have an accuracy slightly lesser than VGG16 but high enough to be suitable for image classification
```sh
Test accuracy: 85.56%
Avg classwise AUC of ROC curve:  0.9690683048083327
```
