import pickle
from abc import ABC, abstractmethod


class Model(ABC):
    """
    Classe mère abstraite : tous les modèles IA doivent respecter
    la même interface (abstraction).
    """

    @abstractmethod
    def train_step(self, X, y, optimizer):
        """Une itération d'apprentissage. Retourne la perte (loss)."""
        pass

    @abstractmethod
    def predict(self, x):
        """Retourne la probabilité d'infection (entre 0 et 1)."""
        pass

    # Méthodes communes héritées par tous les modèles (héritage)
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
