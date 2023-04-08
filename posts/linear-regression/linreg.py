import numpy as np

def pad(X):
    return np.append(X, np.ones((X.shape[0], 1)), 1)

class LinearRegression:
    def __init__(self):
        self.w = None
        self.score_history = []
        
    def linear_regression_gradient(w, X, two_alpha_X_transpose, y):
        return two_alpha_X_transpose * (np.dot(X, w) - y)

    def loss(self, X, y):
        y_pred = np.predict(X)
        return np.linalg.norm(y_pred - y) ** 2

    def predict(self, X):
        if X.shape[1] == len(self.w) - 1: X = pad(X)
        return np.dot(X, self.w)

    def score(self, X, y, denominator=None):
        y_pred = self.predict(X)
        numerator = np.linalg.norm(y_pred - y) ** 2

        if denominator is None:
            y_mean = sum(y) / len(y)
            denominator = np.linalg.norm(y_mean - y) ** 2

        return 1 - (numerator / denominator)

    def set_w(self, features, initial_w=None, range_min= -5, range_max=5):
        if initial_w is None:
            self.w = ((2*range_max) * np.random.random(features)) - range_min
        elif len(initial_w) != features:
            raise Exception(f"Number of initial weights passed does not match number of features: {len(initial_w)}!={features}")
        else:
            self.w = initial_w
    
    def fit_analytical(self, X, y):
        X = pad(X)
        X_transpose = X.T
        self.w = np.dot(np.linalg.inv(np.dot(X_transpose, X)), np.dot(X_transpose, y))

    def fit_gradient(self, X, y, w=None, alpha=.01, max_steps=100):
        X = pad(X)
        X_transpose = X.T
        
        P = np.dot(X_transpose, X)
        q = np.dot(X_transpose, y)
        gradient = lambda w: 2 * (np.dot(P, w) - q)
        
        # for calculating the score
        y_mean = sum(y) / len(y)
        denominator = np.linalg.norm(y_mean - y) ** 2

        self.set_w(initial_w=w, features=X.shape[1])
        for _ in range(max_steps):
            #adjust weights
            self.w = self.w - alpha * gradient(self.w)
            self.score_history.append(self.score(X, y, denominator))
