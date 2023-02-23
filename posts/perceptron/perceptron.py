import numpy as np

class Perceptron:
    def __init__(self):
        """Basic init function to initialize variables."""
        self.w = None
        self.history = []

    def fit(self, X, y, steps=1000):
        """Train perceptron.

        X -- multi-dimensional numpy array
        y -- labels corresponding to X
        steps -- max steps to run before stopping (default 1000)
        """
        n = X.shape[0]
        X_ = np.append(X, np.ones((n, 1)), 1)
        y_ = (2 * y) - 1
        self.w = np.random.randint(-6, 6, X_.shape[1])
        for _ in range(steps):
            i = np.random.randint(n)
            # y_pred = np.sign(np.dot(self.w, X_[i]))  # make prediction
            # false_check = (y_[i] * y_pred) < 0  # check if prediction is wrong
            # self.w = self.w + false_check * (y_[i] * X_[i])  # update if wrong
            self.w = self.w + ((y_[i] * np.dot(self.w, X_[i])) < 0) * (y_[i] * X_[i])
            self.history.append(score := self.score(X, y))
            if score == 1:
                return
        return

    def predict(self, X):
        """Make predictions.

        X -- multi-dimensional numpy array
        """
        if self.w is None:
            raise Exception("Perceptron not yet fit!")
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        return (np.dot(self.w, X_.T) > 0).astype(int)

    def score(self, X, y):
        """Calculate accuracy.

        X -- multi-dimensional numpy array
        y -- labels corresponding to X
        """
        if self.w is None:
            raise Exception("Perceptron not yet fit!")
        y_preds = self.predict(X)
        return sum(y_preds == y) / len(y)
