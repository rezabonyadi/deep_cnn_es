import numpy as np
from keras.models import Sequential
from sklearn import metrics

class DeepCnnEs:
    def __init__(self, deep_model: Sequential, classifier, is_classification=True, iterations=1000, batch_size=128):
        '''
        Initializes the deep model, the classifier/regression method, and maximum number of iterations.

        :param deep_model:
        :param classifier:
        :param iterations:
        :param batch_size
        '''

        self.deep_model = deep_model
        self.classifier = classifier
        self.is_classification = self
        self.iterations = iterations
        self.batch_size = batch_size

    def objective_(self, w: np.ndarray, X: np.ndarray, y: np.ndarray):
        '''
        This the objective to the evolutionary method.

        :param w: The candidate solution.
        :param X: The set of input instances.
        :param y: The labels of the instances (can be classes or values as regression).

        :return: A fitness values.
        '''

        self.update_deep_model_(w)
        X_transformed = self.deep_model.predict(X)
        self.classifier.fit(X_transformed, y)
        y_hat = self.classifier.predict(X_transformed)
        if(self.is_classification):
            return 1.0-metrics.accuracy_score(y, y_hat)
        else: # Regression
            return metrics.mean_squared_error(y, y_hat)

    def update_deep_model_(self, w: np.ndarray):
        '''
        The method gets a deep model, set w for its weights, and returns the modified model.

        :param model: the deep model
        :param w: weights

        :return: Modified model
        '''

        w_index = 0
        weights = self.deep_model.get_weights()
        prepared_weights = list()

        for weight in weights:
            shape = weight.shape
            size = np.prod(shape)
            prepared_weights.append(np.reshape(w[w_index: size+w_index], shape))
            w_index = w_index + size

        self.deep_model.set_weights(prepared_weights)

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def set_model(self, model):
        self.model = model

    def set_classifier(self, classifier):
        self.classifier = classifier


