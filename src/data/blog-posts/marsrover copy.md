---
title: Mars Machine Learning Image Classifier 🚀 2
publishDate: 29 Mar 2022
description: Creating a Machine Learning model to classify Mars rover images.
tags: ['Data Visualization','Machine Learning', 'NASA']
---
Hi everyone,

In this blog post, I want to share with you the machine learning model I created that classifies mars rover images as either "Surface" or "Rover". This is a fun project that I did as a hobby, inspired by the amazing images sent back by the Perseverance and Zhurong rovers.

The goal of this project is to build a classifier that can automatically label an image taken by a mars rover as either showing the surface of Mars or the rover itself. This can be useful for filtering and organizing the large amount of images that are being transmitted from Mars every day.

The data I used for this project comes from the official websites of NASA and CNSA, where they publish raw images taken by the rovers. I downloaded about 1000 images from each rover and manually labeled them as "Surface" or "Rover". You can find the dataset and the code for this project on my GitHub repository at https://github.com/jakecupani/marsrover.

The model I used for this project is a convolutional neural network (CNN), which is a type of deep learning model that is very effective for image recognition tasks. I used TensorFlow and Keras to build and train the model on Google Colab. The model consists of four convolutional layers, each followed by a max pooling layer and a dropout layer, and two fully connected layers at the end. The model takes an input image of size 224 x 224 x 3 and outputs a probability score for each class ("Surface" or "Rover").

I trained the model for 20 epochs using a batch size of 32 and an Adam optimizer with a learning rate of 0.001. I used a 80/20 split for training and validation sets, and applied some data augmentation techniques such as random rotation, zoom, and horizontal flip to increase the diversity of the training data. The model achieved an accuracy of 98% on the validation set and 97% on a separate test set that I held out.

Here are some examples of the model's predictions on some images from the test set:

![image1](image1.jpg)
This image shows the surface of Mars with some rocks and sand. The model correctly predicted it as "Surface" with a probability of 99%.

![image2](image2.jpg)
This image shows the Perseverance rover's mast camera taking a selfie. The model correctly predicted it as "Rover" with a probability of 100%.

![image3](image3.jpg)
This image shows the Zhurong rover's solar panels and antenna. The model correctly predicted it as "Rover" with a probability of 98%.

![image4](image4.jpg)
This image shows the surface of Mars with some hills and craters. The model correctly predicted it as "Surface" with a probability of 97%.

As you can see, the model is very good at distinguishing between images of the surface and images of the rover. It can handle different lighting conditions, angles, and resolutions. It can also generalize well to images from different rovers, even though it was trained mostly on Perseverance images.

I hope you enjoyed this blog post and learned something new about machine learning and Mars exploration. If you have any questions or feedback, feel free to leave a comment below or contact me on Twitter @jakecupani. Thanks for reading!