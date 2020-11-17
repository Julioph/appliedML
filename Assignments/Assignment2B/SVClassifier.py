
import numpy as np
from sklearn.base import BaseEstimator


class LinearClassifier(BaseEstimator):

    def decision_function(self, X):
        return X.dot(self.w)

    def predict(self, X):
        scores = self.decision_function(X)

        out = np.select([scores >= 0.0, scores < 0.0], [self.positive_class, self.negative_class])
        return out

    def find_classes(self, Y):
        classes = sorted(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")

        self.positive_class = classes[1]
        self.negative_class = classes[0]

    def encode_output(self, Y):
        encoded = np.array([1 if y == self.positive_class else -1 for y in Y])
        return encoded


class SVClassifier(LinearClassifier):

    def __init__(self, n_iter):
        self.n_iter = n_iter

    def fit(self, X, Y, regularization_param):
        """
        Train a linear classifier using the SVC learning algorithm.
        """
        self.find_classes(Y)

        Ye = self.encode_output(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        # start iterations
        t = 0
        for i in range(self.n_iter):

            for x_i, y_i in zip(X, Ye):
                t += 1

                # Calculate steplength
                eta = 1 / (regularization_param * t)
                # Calculate score
                score = x_i.dot(self.w)

                if y_i * score < 1.0:
                    self.w = ((1 - eta * regularization_param) * self.w) + ((eta * y_i) * x_i)
                else:
                    self.w = (1 - eta * regularization_param) * self.w
