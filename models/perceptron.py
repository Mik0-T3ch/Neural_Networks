import numpy as np


class Perceptron:
    def __init__(self, lr=0.01, epochs=100, bias=True, seed=None, verbose=False):
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.bias = bool(bias)
        self.seed = seed
        self.verbose = bool(verbose)

        self.w = None
        self.errors_ = []

    def _ensure_arrays(self, X, y=None):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if y is not None:
            y = np.array(y, dtype=float)
            if y.ndim != 1:
                y = y.reshape(-1)
            return X, y
        return X

    def _add_bias(self, X):
        if not self.bias:
            return X
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])

    def _step(self, z):
        return np.where(z >= 0.0, 1, 0)

    def fit(self, X, y):
        X, y = self._ensure_arrays(X, y)
        Xb = self._add_bias(X)

        unique_labels = np.unique(y)
        if len(unique_labels) == 2 and set(unique_labels) == {-1, 1}:
            y = np.where(y == 1, 1, 0)

        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0.0, 0.01, size=Xb.shape[1])
        self.errors_ = []

        for epoch in range(self.epochs):
            errors = 0
            for xi, yi in zip(Xb, y):
                z = np.dot(xi, self.w)
                y_hat = 1 if z >= 0.0 else 0
                error = yi - y_hat
                if error != 0:
                    self.w += self.lr * error * xi
                    errors += 1

            self.errors_.append(errors)

            if self.verbose:
                print(f"Epoca {epoch + 1}/{self.epochs} | Errores: {errors}")

            if errors == 0:
                break

        return self

    def net_input(self, X):
        if self.w is None:
            raise ValueError("El modelo no ha sido entrenado. Llame a fit() primero.")
        X = self._ensure_arrays(X)
        Xb = self._add_bias(X)
        return np.dot(Xb, self.w)

    def predict(self, X):
        z = self.net_input(X)
        return self._step(z)

    def score(self, X, y):
        X, y = self._ensure_arrays(X, y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
