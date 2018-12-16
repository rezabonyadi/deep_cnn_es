import numpy as np
from keras.models import Sequential
from sklearn import metrics
from EvolutionaryAlgorithms.ES import ES
from EvolutionaryAlgorithms.CMAES import CMAES
from sklearn.model_selection import train_test_split

class DeepCnnEs:
    def __init__(self, deep_model: Sequential, classifier, is_classification=True,
                 iterations=1000, batch_size=128, L0 = 0.0, L1 = 0.0, L2 = 0.0):
        '''
        Initializes the deep model, the classifier/regression method, and maximum number of iterations.

        :param deep_model:
        :param classifier:
        :param iterations:
        :param batch_size
        '''

        self.deep_model = deep_model
        self.classifier = classifier
        self.is_classification = is_classification
        self.iterations = iterations
        self.batch_size = batch_size
        self.best_solution = None
        self.L0 = L0
        self.L1 = L1
        self.L2 = L2
        self.set_shape_size_()

    def set_shape_size_(self):
        self.weights_shape = list()
        self.number_of_variables = 0

        weights = self.deep_model.get_weights()
        for weight in weights:
            shape = weight.shape
            self.weights_shape.append(shape)
            self.number_of_variables = self.number_of_variables + np.prod(shape)

    def objective_(self, w: np.ndarray, X: np.ndarray, y: np.ndarray):
        '''
        This the objective to the evolutionary method.

        :param w: The candidate solution.
        :param X: The set of input instances.
        :param y: The labels of the instances (can be classes or values as regression).

        :return: A fitness values.
        '''
        Xp, yp = self.get_batch_(X, y)

        self.update_deep_model_(w)
        X_transformed = self.deep_model.predict(Xp)
        self.classifier.fit(X_transformed, yp)

        yp = y
        X_transformed = self.deep_model.predict(X)

        y_hat = self.classifier.predict(X_transformed)
        reg = self.get_regularization(w)

        if(self.is_classification):
            score = 1.0 - metrics.accuracy_score(yp, y_hat)
        else: # Regression
            score = metrics.mean_squared_error(yp, y_hat)
        # print(score)
        return score + reg


    def get_batch_(self, X, y):
        perm = np.random.permutation(X.shape[0])
        if self.is_classification:
            s = X.shape[0]
            p = self.batch_size/s
            Xp, _, yp, _ = train_test_split(X, y, test_size=1.0-p, shuffle=True, stratify=y)
        else:
            yp = y[perm[0:self.batch_size]]
            Xp = X[perm[0:self.batch_size], :, :, :]
        # print(np.unique(yp).size)
        return Xp, yp

    def get_regularization(self, w: np.ndarray):
        penalty = 0
        if self.L0 > 0:
            penalty = self.L0 * np.count_nonzero(w) / w.size
        if self.L1 > 0:
            penalty = penalty + self.L1 * np.sum(np.abs(w)) / w.size
        if self.L2 > 0:
            penalty = penalty + self.L2 * np.sum(np.dot(w, w.T)) / w.size

        return penalty

    def update_deep_model_(self, w: np.ndarray):
        '''
        The method gets a deep model, set w for its weights, and returns the modified model.

        :param model: the deep model
        :param w: weights

        :return: Modified model
        '''

        w_index = 0
        # weights = self.deep_model.get_weights()
        prepared_weights = list()

        for shape in self.weights_shape:
            size = np.prod(shape)
            prepared_weights.append(np.reshape(w[w_index: size+w_index], shape))
            w_index = w_index + size

        self.deep_model.set_weights(prepared_weights)

    def fit(self, X, y):
        # es = CMAES(self.number_of_variables * [0], 1)
        es = ES(self.number_of_variables * [0], 1)

        es.optimize(self.objective_, iterations=self.iterations,
                    args=(X, y), verb_disp=1)

        # Update the deep model
        res = es.result()
        self.best_solution = res
        self.update_deep_model_(res[0])

        # Update the classifier
        X_transformed = self.deep_model.predict(X)
        self.classifier.fit(X_transformed, y)

    def predict(self, X):
        X_transformed = self.deep_model.predict(X)
        y_hat = self.classifier.predict(X_transformed)
        return y_hat

    def set_model(self, model):
        self.model = model

    def set_classifier(self, classifier):
        self.classifier = classifier


