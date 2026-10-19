import csv
import random


class Dataset:
    """Charge les données patients et les sépare en entraînement / test."""

    FEATURES = ["ecart_temperature", "fievre", "toux", "fatigue"]
    TARGET = "infecte"

    def __init__(self, path):
        self.X = []
        self.y = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.X.append([float(row[c]) for c in self.FEATURES])
                self.y.append(int(row[self.TARGET]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def split(self, test_ratio=0.2, seed=42):
        """Mélange puis sépare : 80 % entraînement, 20 % test."""
        idx = list(range(len(self)))
        random.Random(seed).shuffle(idx)
        n_test = int(len(self) * test_ratio)
        test, train = idx[:n_test], idx[n_test:]
        return ([self.X[i] for i in train], [self.X[i] for i in test],
                [self.y[i] for i in train], [self.y[i] for i in test])
