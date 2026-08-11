import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Add bias column
    X = np.insert(X, 0, 1, axis=1)   # shape: (n_samples, n_features+1)
    weights = np.zeros(X.shape[1])   # initialize weights
    
    for _ in range(steps):
        y_hat = _sigmoid(np.dot(X, weights))  # predictions
        gradient = np.dot(X.T, (y_hat - y)) / X.shape[0]  # gradient
        weights -= lr * gradient  # update
    
    return weights[1:], weights[0]  # return (w, b)
