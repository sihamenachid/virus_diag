import math

from core.model import Model


def sigmoid(z):
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


class LogisticRegression(Model):
    """Régression logistique (hérite de Model)."""

    def __init__(self, n_features):
        self.__weights = [0.0] * n_features   # encapsulation : paramètres protégés
        self.__bias = 0.0

    def predict(self, x):
        z = sum(w * xi for w, xi in zip(self.__weights, x)) + self.__bias
        return sigmoid(z)

    def train_step(self, X, y, optimizer):
        n = len(X)
        grad_w = [0.0] * len(self.__weights)
        grad_b = 0.0
        loss = 0.0
        for x, target in zip(X, y):
            p = self.predict(x)
            error = p - target
            for j in range(len(grad_w)):
                grad_w[j] += error * x[j] / n
            grad_b += error / n
            loss -= (target * math.log(p + 1e-9) + (1 - target) * math.log(1 - p + 1e-9)) / n
        self.__weights = [optimizer.update(w, g) for w, g in zip(self.__weights, grad_w)]
        self.__bias = optimizer.update(self.__bias, grad_b)
        return loss
