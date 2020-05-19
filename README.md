# Facial_Recognition using Transfer Learning

Transfer Learning:

Transfer Learning is like using old experience and learning new things on top of it i.e. we train a pre-trained model with some new data which can be used for predicting some new things and we don't have to provide it with huge datasets and it doesn't require huge computational power as we are training a pre-trained model

Aim:

My aim is to train a pre-trained model for facial recognition by providing it with some new data of human faces.

Requirements:

Libraries:TensorFlow, Keras, opencv, pillow

Files: Haarcascade frontalface, datasets.

Explanation:

Firstly we have to collect our data and for doing so I have used opencv library for capturing images and then I have cropped the images and kept only that part of the image that has a human face in it and for detecting faces in the image I have used Haarcascade model.

Next we have to load our model and here i have used VGG16 model which can be loaded using keras.applications module. Vgg16 model has 13 CRP(Convolution, RELU, Pooling)layers and 3 fully connected layers. We also have to freeze all the layers that has already been trained and then we can add more Layers for training the model for new features. Freezing will help us as some of the hyper-parameters have already been found like weights, number of filters and neurons which already gives us better accuracy.

Next we have to add some new layers so that we can train our model with new features. Here I have added 3 Dense layers and 1 output layer for training our model with new data.

Next we have to load our datasets. We have to provide the locations of both training and testing/validating datasets. As we don't have large amount of data we can use concept of image augmentation(changing the images by rotating,zooming or flipping them.) Image augmentation will increase the number of images thus increasing the dataset.

Next we have to train our model. Firstly we have to compile our model providing it with the loss function, optimizer with its learning rate ,metrics function for accuracy and then we have to fit our data inside the model providing it with the number of epochs.

After just 5 epochs the model has given testing accuracy around 94% and validating accuracy 100%. Although I have provided less data it has given us such great accuracy. This model doesn't require huge computational power because it has already been trained meaning it has already founded the best hyper-parameters and it just have to use those hyper-parameters for finding the features in the image and then it will be able to predict.

Then finally we have to save our model and then provide randomly 10 images for testing the model.
