# Deep-Learning
@ Author - Sandip Dutta

## About
This repository is for storing codes for various deep learning codes written by me. Language used is Python. 

## Contents
* neural_network_from_scratch.py
* churn-imbalanced-multiple-models-best-89-5.ipynb

## Description
__1. neural_network_from_scratch.py__ :

Libraries used: 
* __PyTorch__

It is a Python program to implement a simple neural network on a model using basics of PyTorch. Just the basic functions of PyTorch were used.
Gradient Descent and Backpropagation was implemented from scratch. 

__2. churn-imbalanced-multiple-models-best-89-5.ipynb__:

Libraries used:
* __Numpy__
* __Pandas__
* __Matplotlib__
* __Seaborn__
* __SkLearn__
* __Imblearn__
* __XGBoost__
* __Tensorflow and Keras__

It is an .ipynb(ipython notebook) to demonstrate whether a customer would leave a company or not. The data was read using __Pandas__. Visulalisation was done with the help of __seaborn and Matplotlib__. Data was modified using __Numpy__.

The dataset has about 8000 samples of one class and 2000 samples of another class. This indicates that there was a huge imbalance in the dataset. This was fixed using __SMOTE__ from __imblearn__ library.

Various models were applied on the dataset like:
* Logistic Regression
* Extra Trees Classifier
* Random Forest Classifier
* XgBoost Classifier ( gave the best accuracy of 89.5%)
* Artificial Neural Network 




