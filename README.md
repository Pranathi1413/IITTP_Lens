# IITTP_Lens

* ML Model to identify (to the best of our model's accuracy) the hang-out spots at IIT Tirupati Campus. 
* It uses a VGG16, a CNN model, as of now, and is limited to predicting a class of 27 places.

* It is currently deployed at: https://iittp-lens.herokuapp.com/


## The Classes
```
0:'TC-1',  1:'Classroom Complex',  2:'CS Lab', 3:'Cricket/Football ground', 4:'Front', 5:'Girls hostel', 6:'Guest house', 7:'Gym', 8:'Health center', 9:'Hostel', 10:'Indoor Stadium', 11:'Lab', 12:'Library', 13:'Mess', 14:'OAT steps', 15:'Outdoor courts', 16:'Parking lot', 17:'Roads', 18:'Classroom Complex', 19:'Classroom Complex', 20:'Hostels', 21:'Hostel', 22:'Indoor Stadium', 23:'Lab', 24:'Library', 25:'OAT', 26:'TC-22'
```

NOTE: THE PR AND ROC CURVES OBTAINED ON THE FOLLWING MODELS ARE PLACED IN 'observations/' FOLDER.

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

### Random Forest
>This is a Supervised Machine Learning algorithm, widely used in Classification and Regression problems. It uses the ensamble technique and builds decision tree for each sample and takes majority vote in case of classification. 

Here, test accuracy was very less comapred to CNN model accuracies. Thus, it failed to classify majority of the images to their respective classes.
```sh
Training Accuracy: 100.0%
Test Accuracy: 56.67%
Classwise average ROC-AUC Score: 0.9629629629629629
```
### K Nearest Neighbors
>This is also a Supervised Machine Learning algorithm that is most commonly used in Classification problems. As the name suggests it considers K Nearest Neighbors (Data points) to predict the class or continuous value for the new Datapoint. 

The model takes K as an input,after few trials it was found that K = 1 gives the best test accuracy for this problem. This makes sense because the data points from different classes differe widely. 
```sh
Test Accuracy: 51.11%
```
That said, this accuracy is not the best as compared to other models'. From the PR and ROC curves it can be observed that the classwise avg AUC is as around 0.6 which doesn't make it a good classifier.

### Stochastic Gradient Descent
>This estimator implements regularized linear models(SVM, logistic regression, etc.) with SGD learning i.e the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing learning rate.
We tried this model to see how linear models perform on this problem. As per the test accuracy, this model is not correct for this multi classification problem.
```sh
Test Accuracy: 21.48%
```
Also it can be observed that the classwise avg AUC is less than 0.5 from the PR and ROC curves. Thus, this classifier doesn't work best for this problem.
