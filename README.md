# Binary Classification Neural Network - MNIST Dataset

This project implements a simple binary classification neural network to distinguish between the digits **0** and **1** from the MNIST dataset. The model achieves excellent accuracy and low loss on both the training and validation sets, demonstrating its effectiveness in the task.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

The project showcases a binary classification model built using TensorFlow and Keras. It includes custom implementations of the dense layer and sequential model functionality to predict the digit labels using NumPy, alongside TensorFlow’s model predictions.

## Dataset

The MNIST dataset is used, but filtered to include only samples for the digits **0** and **1**:
- **Training Set**: Filtered subset of the original training data.
- **Test Set**: Filtered subset of the original test data.

Data is normalized to the range `[0, 1]` and reshaped to a flat vector for input into the neural network.

## Model Architecture

The neural network is built with the following layers:

1. **Input Layer**: 784 units (28 x 28 flattened image input)
2. **Hidden Layer 1**: Dense layer with 128 units and ReLU activation
3. **Hidden Layer 2**: Dense layer with 64 units and ReLU activation
4. **Output Layer**: Dense layer with 1 unit and sigmoid activation for binary classification

## Results

For detailed results, visit the [results directory](https://github.com/Torajabu/BINARY-CLASSIFICATION---NEURAL-NET---MNIST-DATASET/tree/main/BINARY%20LASSIFICATION%20RESULTS%20NEURAL%20NET).

### Metrics
- **Accuracy**: The model achieves high accuracy on both training and validation sets.
- **Loss**: Low binary cross-entropy loss indicates the model performs well in distinguishing between the digits **0** and **1**.


## Usage

### Prerequisites

Ensure the following Python libraries are installed:
- TensorFlow
- NumPy
- Matplotlib

### Steps to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Torajabu/BINARY-CLASSIFICATION---NEURAL-NET---MNIST-DATASET.git
   cd BINARY-CLASSIFICATION---NEURAL-NET---MNIST-DATASET
   ```

2. Run the script:
   ```bash
   python binary_classification_mnist.py
   ```

3. Visualize the results:
   - The script will display the results of predictions from both TensorFlow and the custom NumPy implementation.

### Key Functions
- **my_dense**: Custom implementation of a dense layer.
- **my_sequential**: Custom implementation of a sequential model.
- **sigmoid**: Sigmoid activation function.

## Acknowledgments

Special thanks to the creators of the MNIST dataset for providing such an invaluable resource for machine learning and deep learning projects.

