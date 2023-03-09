import numpy as np
from scipy.optimize import minimize


def sigmoid(x):
    return 1/(1 + np.exp(np.negative(x)))

def logistic_loss(theta, X_, y_actual):
    y_pred = np.dot(theta, X_.T)
    sigmoid_y_pred = sigmoid(y_pred)
    return np.multiply(-1*y_actual, np.log(sigmoid_y_pred)) - np.multiply(1 - y_actual, np.log(1 - sigmoid_y_pred))

class KernelLogisticRegression:
    def __init__(self, kernel, **kernel_kwargs):
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs

        self.theta = None

    def predict(self, X):
        if self.theta is None: raise Exception("Perceptron not yet fit!")

        n = X.shape[0]
        X_ = np.append(X, np.ones((n, 1)), 1)
        kernel_X = self.kernel(self.X_train, X_, **self.kernel_kwargs)

        return (np.dot(self.theta, kernel_X) > 0).astype(int)

    def score(self, X, y):
        if self.theta is None: raise Exception("Perceptron not yet fit!")
        y_preds = self.predict(X)
        return sum(y_preds == y) / len(y)


    def fit(self, X, y):
        n = X.shape[0]
        self.theta = ((2*1) * np.random.random(n)) - 1
        self.X_train = np.append(X, np.ones((n, 1)), 1)

        kernel_X = self.kernel(self.X_train, self.X_train, **self.kernel_kwargs)

        # perform the minimization
        result = minimize(lambda theta: np.mean(logistic_loss(theta, kernel_X, y)), x0=self.theta)
        self.theta = result.x
