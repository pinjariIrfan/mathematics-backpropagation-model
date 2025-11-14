"""
Activation Functions Implementation
Common activation functions used in neural networks
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# Test the functions
if __name__ == "__main__":
    test_data = np.array([-1, 0, 1, 2])
    print("Sigmoid:", sigmoid(test_data))
    print("ReLU:", relu(test_data))
    print("Tanh:", tanh(test_data))