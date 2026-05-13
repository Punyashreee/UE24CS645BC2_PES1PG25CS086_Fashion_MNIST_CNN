import numpy as np

# CROSS ENTROPY LOSS

def cross_entropy_loss(y_pred, y_true):

    m = y_true.shape[0]

    epsilon = 1e-12

    y_pred = np.clip(
        y_pred,
        epsilon,
        1. - epsilon
    )

    loss = -np.log(
        y_pred[range(m), y_true]
    )

    return np.sum(loss) / m



# ACCURACY

def accuracy(y_pred, y_true):

    predictions = np.argmax(
        y_pred,
        axis=1
    )

    return np.mean(predictions == y_true)