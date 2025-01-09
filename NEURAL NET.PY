#FOR RESULTS OF THIS NEURAL NETWORK, PLEASE REFER TO https://github.com/Torajabu/BINARY-CLASSIFICATION---NEURAL-NET---MNIST-DATASET/tree/main/BINARY%20LASSIFICATION%20RESULTS%20NEURAL%20NET

#model achieved excellent accuracy and very low loss on both training and validation sets. This indicates that the model is performing well in distinguishing between the digits 0 and 1

import warnings
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Filter the dataset to include only digits 1 and 0
train_filter = np.where((y_train == 0) | (y_train == 1))
test_filter = np.where((y_test == 0) | (y_test == 1))

X_train, y_train = X_train[train_filter], y_train[train_filter]
X_test, y_test = X_test[test_filter], y_test[test_filter]

# Normalize and reshape the data
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# Define the model with the correct input shape
model = Sequential([
    Dense(128, input_shape=(784,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define my_dense function
def my_dense(a_in, W, b, g):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        z_j = np.dot(a_in, W[:, j]) + b[j]
        a_out[j] = g(z_j)
    return a_out

# Define my_sequential function
def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x, W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return a3

# Get weights from the model
W1_tmp, b1_tmp = model.layers[0].get_weights()
W2_tmp, b2_tmp = model.layers[1].get_weights()
W3_tmp, b3_tmp = model.layers[2].get_weights()

# Plot the results
m, n = X_train.shape
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])

for i, ax in enumerate(axes.flat):
    random_index = np.random.randint(m)
    X_random_reshaped = X_train[random_index].reshape((28, 28)).T
    ax.imshow(X_random_reshaped, cmap='gray')
    
    # TensorFlow Prediction
    tf_prediction = model.predict(X_train[random_index].reshape(1, 784))
    tf_yhat = int(tf_prediction.item() >= 0.5)
    
    # Numpy Prediction
    my_prediction = my_sequential(X_train[random_index], W1_tmp, b1_tmp, W2_tmp, b2_tmp, W3_tmp, b3_tmp)
    my_yhat = int(my_prediction.item() >= 0.5)
    
    ax.set_title(f"{y_train[random_index]}, {tf_yhat}, {my_yhat}")
    ax.set_axis_off()

fig.suptitle("Label, yhat Tensorflow, yhat Numpy", fontsize=16)
plt.show()
