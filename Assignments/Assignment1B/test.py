import numpy as np
import pandas as pd

class LinearClassifier(object):

    def __init__(self):
        self.positive_class = 0
        self.negative_class = 1

    def find_classes(self, y):
        """
        Finds the set of output classes in the output part Y of the training set.
        If there are exactly two classes, one of them is associated to positive
        classifier scores, the other one to negative scores. If the number of classes
        is not 2, an error is raised.
        """
        classes = sorted(set(y))
        if len(classes) != 2:
            raise Exception("Thisdoes not seem to be a binary class problem")
        self.positive_class = classes[0]
        self.negative_class = classes[1]

    def predict(self, xtest):
        """
        Predicts the outputs for the inputs X. The inputs are assumed to be stored in
        a matrix, where each row contains the features for one instance.
        """

        scores = np.dot(xtest, self.weights)
        out = np.select([scores > 0.0, scores < 0.0], [self.positive_class, self.negative_class])
        return out


class Percept(LinearClassifier):

    def __init__(self, n_iter):
        self.n_iter = n_iter

    def fit(self, x_train, y_train):

        self.find_classes(y_train)

        n_features = x_train.shape[1]
        self.weights = np.zeros(n_features)

        for i in range(self.n_iter):
            for x, y in zip(x_train, y_train):
                score = x.dot(self.weights)

                if y == self.positive_class and score < 0.0:
                    self.weights += x
                if y == self.negative_class and score > 0.0:
                    self.weights -= x


df = pd.read_csv("Beijing_labeled.csv")