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
        """Calculate the loss.

        w -- weights
        X_ -- multi-dimensional numpy array with ones column added
        y_actual -- labels corresponding to X
        """
        y_pred = np.dot(w, X_.T).clip(-10, 10)
        sigmoid_y_pred = sigmoid(y_pred)
        return np.multiply(-1*y_actual, np.log(sigmoid_y_pred)) - np.multiply(1 - y_actual, np.log(1 - sigmoid_y_pred))

    def logistic_loss_gradient(self, w, X_, y_actual):
        """Calculate the gradient of the loss function.

        w -- weights
        X_ -- multi-dimensional numpy array with ones column added
        y_actual -- labels corresponding to X
        """
        y_pred = np.dot(w, X_.T)
        return np.mean(np.multiply(sigmoid(y_pred) - y_actual, X_.T), axis=1)

    def set_w(self, initial_w, features, range_min= -5, range_max=5):
        """Set weights.

        initial_w -- initial weights, if they are already set, this function will use those values
        features -- number of features of the data
        range_min -- minimun for randomly generated features
        range_max -- maximum for randomly generated features
        """
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

        This function trains logistic regression models with gradient descent.
        It is a very simple training method, with no complex features. It
        simply takes the gradient, multiplies it by alpha, and then subtracts
        that from the weights.
        """
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        self.set_w(initial_w, X_.shape[1])
        for _ in range(max_epochs):
            #adjust weights
            self.w = self.w - alpha * self.logistic_loss_gradient(self.w, X_, y)

            # record values
            if track_w: self.w_history.append(self.w)
            self.loss_history.append(np.mean(self.logistic_loss(self.w, X_, y)))
            self.score_history.append(self.score(X_, y))

    def fit_stochastic(self, X, y, batch_size, momentum=0, initial_w=None, alpha=.1, max_epochs=100):
        """Train algorithm with more options.

        X -- multi-dimensional numpy array
        initial_w -- starting weights
        y -- labels corresponding to X
        batch_size -- batch size
        alpha -- alpha
        max_epochs -- max epochs to run before stopping (default 1000)

        This function trains logistic regression models with gradient descent.
        It has a couple more features that the fit(...) function does.

        It trains stochastically, using a batch size of the users choice,
        meaning it divides the dataset into smaller batches and adjusts all of
        weights based off just that data. In one epoch, it will still run
        through the whole dataset, but it will have adjusted the weights many
        times.

        This function also has the option to train with momentum. This operation
        adds the change in weights from the previous epoch to the current
        weight.
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

                # calculate momentum
                moment = 0
                if momentum and len(self.loss_history) > 1:
                    moment = momentum * np.subtract(self.w, w_prev)

                # record weights before they are adjusted
                w_prev=self.w

                # adjust weights
                self.w = self.w - alpha * self.logistic_loss_gradient(self.w, X_batch, y_batch) + moment
            # record history
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
        """Calculate loss.

        X -- multi-dimensional numpy array
        y -- labels corresponding to X
        """
        if X_.shape[1] != len(self.w):
            raise Exception(f"X does not have correct number of features: {X_.shape[1]}!={len(self.w)}.")
        return np.mean(self.logistic_loss(self.w, X_, y))
