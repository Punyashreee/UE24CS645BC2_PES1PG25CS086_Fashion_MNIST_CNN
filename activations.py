import numpy as np

# ReLU Activation
def relu(x):

    return np.maximum(0, x)



# ReLU Derivative
def relu_derivative(x):

    return (x > 0).astype(float)


# Softmax Activation
def softmax(x):

    # Prevent overflow
    exp_x = np.exp(x - np.max(x))

    return exp_x / np.sum(exp_x, axis=1, keepdims=True)