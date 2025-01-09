MNIST Binary Classification (0 vs 1)
This repository contains a project for binary classification of MNIST digits, specifically distinguishing between digits 0 and 1. The project includes training a neural network using TensorFlow and implementing a custom neural network using Numpy for comparison.

Table of Contents
Introduction

Dataset

Model

Custom Neural Network

Usage

Results

Requirements

Acknowledgments

Introduction
This project aims to classify MNIST digits (0 and 1) using a neural network implemented in TensorFlow and a custom neural network implemented in Numpy. The project provides a visual comparison of the predictions made by both models.

Dataset
The dataset used is the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits. Only digits 0 and 1 are used for this binary classification task.

Model
The TensorFlow model is a simple feedforward neural network with the following architecture:

Input layer with 784 units

Hidden layer 1 with 128 units and ReLU activation

Hidden layer 2 with 64 units and ReLU activation

Output layer with 1 unit and sigmoid activation

Custom Neural Network
A custom neural network is implemented using Numpy to mimic the TensorFlow model. The network uses the sigmoid activation function and consists of three layers.

Usage
To run the project, follow these steps:

Clone this repository.

Install the required libraries listed in the Requirements section.

Run the test.py script.

bash
git clone https://github.com/Torajabu/BINARY-CLASSIFICATION---NEURAL-NET---MNIST-DATASET
cd https://github.com/Torajabu/BINARY-CLASSIFICATION---NEURAL-NET---MNIST-DATASET
pip install -r requirements.txt
python test.py
Results
The model achieves high accuracy in distinguishing between digits 0 and 1. The results are visualized using Matplotlib, showing predictions from both TensorFlow and the custom Numpy implementation.

Requirements
Python 3.x

TensorFlow

Numpy

Matplotlib

Install the requirements using pip:

bash
pip install tensorflow numpy matplotlib
Acknowledgments
The MNIST dataset is provided by Yann LeCun's website.

TensorFlow and Numpy libraries for machine learning and numerical computations.
