from utils.metrics import accuracy, precision, recall, f1_score


class Evaluator:
    """Mesure les performances du modèle (accuracy, precision, recall, F1)."""

    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def evaluate(self, X, y):
        y_pred = [1 if self.model.predict(x) >= self.threshold else 0 for x in X]
        return {
            "accuracy": accuracy(y, y_pred),
            "precision": precision(y, y_pred),
            "recall": recall(y, y_pred),
            "f1": f1_score(y, y_pred),
        }
