# ACSE 4.4 - Machine Learning Miniproject
# The Identification Game

The purpose of the Identification Game is to classify 64 by 64 colour (RGB) images into one of 200 categories. This software is capable of such a task and contains a number of machine learning models which are able to classify images. The models are:

* LeNet5
* VGG16
* GoogleNet
* ResNet
* Wide ResNet

All information available here: https://www.kaggle.com/c/acse-miniproject/overview

As well as allowing the user to train a model, a pre-trained model can be loaded 

## Installation Guide

To install the software, execute the below command in a folder to clone into that folder: 

```
git clone https://github.com/acse-2019/acse4-4-divergence.git
```

## User Instructions

## Model training
We show how we train our best single model, Wide ResNet50, in the model_training.ipynb notebook. For training the other models, simply change the line
```
model = dm.FineTuningClassifier(dm.FineTuningModel.wide_resnet50)
```
to use another available model, listed in the enum `dm.FineTuningModel`. The notebook generates a csv file with predictions. The data needs to be mounted as a zip file from google drive, using the following file:

https://drive.google.com/file/d/14otNyLOe0fccF-vny3t4i2L4IA0lP_85/view?usp=sharing

Or you can download the data from kaggle and zip the folder.

## Loading pre-trained models

No time left to produce a specific notebook for that. So:
* Download the pth files from this folder: https://drive.google.com/open?id=13xlBQxd80tuSh4v-UjSTYi5uhdPIudWU
* Unzip the zip file.
* Use model_training.ipynb: On this line: `model = dm.FineTuningClassifier(dm.FineTuningModel.wide_resnet50)`, choose the right classifier from the enum, and then provide the state dict (loaded with `torch.load()`) to the function:

```python
model = dm.FineTuningClassifier(dm.FineTuningModel.wide_resnet50, state_dict=torch.load("path/to/state_dict.pth"))
```

## 
The ``` ensemble ``` file can be loaded from the ``` divergence ``` folder which allows a label to be assigned to a test image based on the predictions of several models which each have a vote to determine the assignment.

Files needed to run script:
https://drive.google.com/drive/folders/1u16HHh8PUKmzB01Xx4RNurPbo8GEU3Vb?usp=sharing
