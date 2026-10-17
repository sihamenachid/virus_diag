class Trainer:
    """Entraîne le modèle (boucle d'apprentissage)."""

    def __init__(self, model, optimizer, epochs=100):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs

    def fit(self, X, y, verbose=True):
        for epoch in range(1, self.epochs + 1):
            loss = self.model.train_step(X, y, self.optimizer)   # polymorphisme
            if verbose and epoch % (self.epochs // 4) == 0:
                print(f"   epoch {epoch:4d} | loss = {loss:.4f}")
        return self.model
