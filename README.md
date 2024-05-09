# Handwritten Digit Recognition using Neural Networks

## Description

This repository contains Python scripts for building a neural network model to recognize handwritten digits using the MNIST dataset. The dataset consists of 28x28 pixel grayscale images of handwritten digits (0 to 9). The neural network model is trained on the training set and evaluated on the testing set to measure its accuracy in digit recognition.

## Data Loading and Preprocessing

The MNIST dataset is loaded using TensorFlow's built-in dataset loader. The training and testing data are preprocessed by reshaping the images into 1D arrays, normalizing pixel values, and converting labels into one-hot encoded vectors.

## Model Architecture

The neural network model consists of two hidden layers with ReLU activation functions followed by an output layer with softmax activation. The input layer has 784 nodes corresponding to the flattened image pixels, and the output layer has 10 nodes representing the digits 0 through 9.

## Model Training and Evaluation

The model is trained using stochastic gradient descent with the Adam optimizer and cross-entropy loss function. The training loop iterates over multiple epochs, with each epoch consisting of batches of training data. After training, the model's accuracy is evaluated on the testing set.

## Results

The training process displays the cost per epoch, indicating the average loss over the training data. After training, the model's accuracy is calculated by comparing predicted labels with true labels on the testing set. The accuracy metric shows the model's performance in recognizing handwritten digits.

## Conclusion

The neural network model demonstrates effective digit recognition on the MNIST dataset, achieving high accuracy in identifying handwritten digits. Further optimization and experimentation with different architectures or hyperparameters could potentially improve the model's performance.
