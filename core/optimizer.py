class GradientDescent:
    """Optimiseur : descente de gradient."""

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, param, grad):
        return param - self.learning_rate * grad
