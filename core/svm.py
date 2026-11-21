from core.model import Model
from core.logistic_regression import sigmoid


class SVM(Model):
    """SVM linéaire (Support Vector Machine), entraîné par descente de gradient (hérite de Model)."""

    def __init__(self, n_features):
        self.__weights = [0.0] * n_features   # encapsulation : paramètres protégés
        self.__bias = 0.0

    def __decision(self, x):
        return sum(w * xi for w, xi in zip(self.__weights, x)) + self.__bias

    def predict(self, x):
        # score de décision transformé en probabilité (>= 0.5 si le patient est du côté "infecté")
        return sigmoid(self.__decision(x))

    def train_step(self, X, y, optimizer):
        n = len(X)
        grad_w = [0.01 * w for w in self.__weights]   # régularisation
        grad_b = 0.0
        loss = 0.0
        for x, target in zip(X, y):
            label = 1 if target == 1 else -1          # l'SVM utilise les classes -1 / +1
            margin = label * self.__decision(x)
            if margin < 1:                             # hinge loss : max(0, 1 - marge)
                loss += (1 - margin) / n
                for j in range(len(grad_w)):
                    grad_w[j] -= label * x[j] / n
                grad_b -= label / n
        self.__weights = [optimizer.update(w, g) for w, g in zip(self.__weights, grad_w)]
        self.__bias = optimizer.update(self.__bias, grad_b)
        return loss
