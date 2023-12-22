![TF](https://github.com/Mubazir-Bangkit-2023/mubazir-machine-learning/assets/95016158/cf4884a9-2a4d-4148-a5a7-24a57c009da0)

# Food Image Classification
Developing a deep learning model for food image classification using Tensorflow: Creating a Convolutional Neural Network (CNN) model to categorize fruits and vegetables with Tensorflow.

## Project Overview
The objective of this project is to construct a model capable of precisely categorizing images of fruits and vegetables into predetermined classes. This model has the potential to be incorporated into applications, enabling the automatic identification and classification of the fruit or vegetable type and determining its freshness based on images uploaded by users.

## Dataset
The dataset employed in this project comprises images showcasing a variety of fruits and vegetables systematically arranged into distinct categories. Every image is annotated with the specific category of the fruit or vegetable and its corresponding freshness classification.
- https://drive.google.com/drive/folders/1ebQa96MrdZYi1mj4PJUZbrDSyXWQ0xd-?usp=sharing

## Feature
- Data Augmentation 
- CNN (Convolutional Neural Networks)
- Transfer Learning (Mobinenetv2)

## Requirements 
- Tensorflow 
- Matplothlib
- Numpy
- Pillow
- Scikit-learn

## Documentation
- Conduct research related to machine learning models
- Collect fresh and rotten fruits and vegetables datasets 
- Split the dataset into 70% for the training set and 30% for the validation set
- Create a machine learning model with CNN (Convolutional Neural Network) using MobileNetV2 architecture
- Train the model 
- Evaluate the machine learning model
- Testing the model by uploading new test data that the model has never seen before
- Deploy the model using Fast Api

## Results
The model achieved a test accuracy of 97% on the test dataset. The training and validation loss/accuracy plots can be found here.
![acc](https://github.com/Mubazir-Bangkit-2023/mubazir-machine-learning/assets/95016158/6f2c7efe-5c95-42ad-9b88-568b92606d11)

![loss](https://github.com/Mubazir-Bangkit-2023/mubazir-machine-learning/assets/95016158/64495ff1-56f3-4b27-a2d6-4597ce07d845)


## Future Work
- Expand the data set to include a wider variety of fruits and vegetables
- Explore advanced architectures and methodologies in experiments to enhance accuracy further.
