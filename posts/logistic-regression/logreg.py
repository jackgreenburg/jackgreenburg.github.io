import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(np.negative(x)))

class LogisticRegression:
    def __init__(self):
        """Basic init function to initialize variables."""
        self.w = None
        self.loss_history = []
        self.score_history = []
        self.w_history = []

    def logistic_loss(self, w, X_, y_actual):
        y_pred = np.dot(w, X_.T).clip(-10, 10)
        sigmoid_y_pred = sigmoid(y_pred)
        return np.multiply(-1*y_actual, np.log(sigmoid_y_pred)) - np.multiply(1 - y_actual, np.log(1 - sigmoid_y_pred))

    def logistic_loss_gradient(self, w, X_, y_actual):
        y_pred = np.dot(w, X_.T)
        return np.mean(np.multiply(sigmoid(y_pred) - y_actual, X_.T), axis=1)

    def set_w(self, initial_w, features, range_min= -5, range_max=5):
        if initial_w is None:
            self.w = ((2*range_max) * np.random.random(features)) - range_min
        else:
            self.w = initial_w

    def fit(self, X, y, initial_w=None, alpha=.1, max_epochs=100, track_w=False):
        """Train algorithm.

        X -- multi-dimensional numpy array
        y -- labels corresponding to X
        initial_w -- starting weights
        alpha -- alpha
        max_epochs -- max epochs to run before stopping (default 100)
        """
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)

        self.set_w(initial_w, X_.shape[1])

        self.history = []
        for _ in range(max_epochs):
            y_pred = self.predict(X_)

            self.w = self.w - alpha * self.logistic_loss_gradient(self.w, X_, y)

            if track_w: self.w_history.append(self.w)
            self.loss_history.append(np.mean(self.logistic_loss(self.w, X_, y)))
            self.score_history.append(self.score(X_, y))

    def fit_stochastic(self, X, y, batch_size, momentum=0, initial_w=None, alpha=.1, max_epochs=100):
        """Train algorithm.

        X -- multi-dimensional numpy array
        initial_w -- starting weights
        y -- labels corresponding to X
        batch_size
        alpha -- alpha
        max_epochs -- max epochs to run before stopping (default 1000)
        """
        n = X.shape[0]
        X_ = np.append(X, np.ones((n, 1)), 1)
        self.set_w(initial_w, X_.shape[1])

        w_prev=None
        for _ in range(max_epochs):

            order = np.arange(n)
            np.random.shuffle(order)
            for batch in np.array_split(order, n // batch_size + 1):
                X_batch = X_[batch,:]
                y_batch = y[batch]

                moment = 0
                if momentum and len(self.loss_history) > 1:
                    moment = momentum * np.subtract(self.w, w_prev)
                    # moment = momentum * np.add(self.w, w_prev)
                w_prev=self.w
                y_pred = self.predict(X_batch)
                self.w = self.w - alpha * self.logistic_loss_gradient(self.w, X_batch, y_batch) + moment
            y_pred = self.predict(X_)
            self.loss_history.append(np.mean(self.logistic_loss(self.w, X_batch, y_batch)))
            self.score_history.append(self.score(X_, y))

    def predict(self, X_):
        """Make predictions.

        X -- multi-dimensional numpy array
        """
        if self.w is None:
            raise Exception("Not yet fit!")
        if X_.shape[1] != len(self.w):
            raise Exception(f"X does not have correct number of features: {X_.shape[1]}!={len(self.w)}.")

        return (np.dot(self.w, X_.T) > 0).astype(int)

    def score(self, X_, y):
        """Calculate accuracy.

        X -- multi-dimensional numpy array
        y -- labels corresponding to X
        """
        if self.w is None:
            raise Exception("Not yet fit!")
        if X_.shape[1] != len(self.w):
            raise Exception(f"X does not have correct number of features: {X_.shape[1]}!={len(self.w)}.")
        y_preds = self.predict(X_)
        return np.mean(y_preds == y)

    def loss(self, X_, y):
        if X_.shape[1] != len(self.w):
            raise Exception(f"X does not have correct number of features: {X_.shape[1]}!={len(self.w)}.")
        return np.mean(self.logistic_loss(self.w, X_, y))
