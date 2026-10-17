class Normalizer:
    """Nettoyage / normalisation des données : centre-réduit chaque colonne."""

    def fit(self, X):
        n = len(X)
        cols = list(zip(*X))
        self.means = [sum(c) / n for c in cols]
        self.stds = [max((sum((v - m) ** 2 for v in c) / n) ** 0.5, 1e-9)
                     for c, m in zip(cols, self.means)]
        return self

    def transform(self, X):
        return [[(v - m) / s for v, m, s in zip(x, self.means, self.stds)] for x in X]

    def transform_one(self, x):
        return self.transform([x])[0]
