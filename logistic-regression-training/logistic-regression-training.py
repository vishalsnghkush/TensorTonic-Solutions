import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X=np.insert(X,0,1,axis=1)
    weights = np.zeros(X.shape[1])
    for i in range(steps):
        y_hat = _sigmoid(np.dot(X, weights))
        slope = (np.dot((y-y_hat),X))/X.shape[0]
        weights = weights + lr * slope
    return weights[1:],weights[0]