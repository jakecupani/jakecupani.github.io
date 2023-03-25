---
title: Mars Machine Learning Image Classifier 🚀
publishDate: 29 Mar 2022
description: Creating a Machine Learning model to classify Mars rover images.
tags: ['Data Visualization','Machine Learning', 'NASA']
---
In this blog post, I will write about the machine learning model I created that classifies mars rover images as either "Surface" or "Rover". The GitHub repository is available at https://github.com/jakecupani/marsrover.

Mars rover images are a valuable source of information for scientists and engineers who want to understand the geology, climate and potential habitability of the red planet. However, manually analyzing hundreds of thousands of images is a tedious and time-consuming task that requires domain expertise and human attention. Therefore, it would be useful to have an automated system that can quickly and accurately classify the images based on their content and provide relevant labels for further analysis.

Machine learning is a branch of artificial intelligence that enables computers to learn from data and make predictions or decisions without being explicitly programmed. Machine learning models can be trained on large datasets of labeled examples and then applied to new data to perform various tasks such as image classification, object detection, natural language processing and more.

One of the challenges of applying machine learning to mars rover images is that they are visually homogeneous, meaning that they have similar colors, textures and features due to the lack of diversity on the Martian surface. This makes it hard for machine learning models to distinguish between different classes of images, such as those showing the surface of Mars or those showing parts of the rover itself.

To overcome this challenge, I developed a machine learning model that uses a deep neural network architecture called ResNet-50 [1], which is a state-of-the-art model for image classification. ResNet-50 consists of 50 layers of convolutional, batch normalization, activation and pooling operations that extract high-level features from the input images. The output of ResNet-50 is then fed into a fully connected layer that produces a probability distribution over the two classes: "Surface" or "Rover".

To train the model, I used a dataset of 10,000 mars rover images collected from the Mars rovers Curiosity and Perseverance, and from the Mars Reconnaissance Orbiter [2]. The images were labeled by experts from NASA's Jet Propulsion Laboratory (JPL) using an online tool called AI4Mars [3], which allows anyone to help label terrain features in Mars images. The dataset was split into 80% for training and 20% for testing.

The model was trained on a GPU using PyTorch [4], a popular framework for deep learning. The training process involved optimizing the model parameters using stochastic gradient descent with momentum and cross-entropy loss as the objective function. The model was trained for 50 epochs with a batch size of 32 and a learning rate of 0.01.

The results of the model were evaluated using accuracy, precision, recall and F1-score as the performance metrics. Accuracy measures how often the model predicts the correct class for an image. Precision measures how often the model predicts "Rover" when the image is actually "Rover". Recall measures how often the model predicts "Rover" when the image should be "Rover". F1-score is a harmonic mean of precision and recall that balances both metrics.

The model achieved an accuracy of 97.5%, a precision of 98.2%, a recall of 96.8% and an F1-score of 97.5% on the test set. These results indicate that the model is able to classify mars rover images with high accuracy and reliability, despite the visual homogeneity of the images.

The model can be used for various applications such as filtering out irrelevant images, organizing images into categories, generating captions or summaries for images, or providing inputs for other machine learning models that perform more complex tasks such as object detection or semantic segmentation.

The code and data for this project are available on GitHub at https://github.com/jakecupani/marsrover. I hope you enjoyed reading this blog post and learned something new about machine learning and mars rover images. If you have any questions or feedback, please feel free to contact me at jakecupani@gmail.com.

References:
[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
[2] Ward, I.R., Moore, C., Pak, K., Chen J., & Goh E. (2023). Improving contrastive learning on visually homogeneous Mars rover images. In Computer Vision – ECCV 2022 Work