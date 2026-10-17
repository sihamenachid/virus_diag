import math
import random

from core.model import Model
from core.logistic_regression import sigmoid


class NeuralNetwork(Model):
    """Réseau de neurones à une couche cachée (hérite de Model)."""

    def __init__(self, n_features, n_hidden=4, seed=0):
        rng = random.Random(seed)
        # encapsulation : poids protégés
        self.__W1 = [[rng.uniform(-0.5, 0.5) for _ in range(n_features)] for _ in range(n_hidden)]
        self.__b1 = [0.0] * n_hidden
        self.__W2 = [rng.uniform(-0.5, 0.5) for _ in range(n_hidden)]
        self.__b2 = 0.0

    def __forward(self, x):
        h = [sigmoid(sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.__W1, self.__b1)]
        out = sigmoid(sum(w * hi for w, hi in zip(self.__W2, h)) + self.__b2)
        return h, out

    def predict(self, x):
        return self.__forward(x)[1]

    def train_step(self, X, y, optimizer):
        loss = 0.0
        for x, target in zip(X, y):
            h, out = self.__forward(x)
            loss -= (target * math.log(out + 1e-9) + (1 - target) * math.log(1 - out + 1e-9)) / len(X)
            # rétropropagation
            d_out = out - target
            d_hidden = [d_out * w2 * hi * (1 - hi) for w2, hi in zip(self.__W2, h)]
            self.__W2 = [optimizer.update(w, d_out * hi) for w, hi in zip(self.__W2, h)]
            self.__b2 = optimizer.update(self.__b2, d_out)
            for i in range(len(self.__W1)):
                self.__W1[i] = [optimizer.update(w, d_hidden[i] * xi) for w, xi in zip(self.__W1[i], x)]
                self.__b1[i] = optimizer.update(self.__b1[i], d_hidden[i])
        return loss
