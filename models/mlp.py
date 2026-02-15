import numpy as np

from utils.activation import get_activation
from utils.loss import get_loss


class MLP:
    def __init__(
        self,
        n_inputs,
        n_hidden=8,
        n_outputs=1,
        lr=0.1,
        epochs=1000,
        hidden_activation="tanh",
        output_activation="sigmoid",
        loss="bce",
        seed=None,
        verbose=False,
    ):
        self.n_inputs = int(n_inputs)
        self.n_hidden = int(n_hidden)
        self.n_outputs = int(n_outputs)

        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = seed
        self.verbose = bool(verbose)

        self.h_act = get_activation(hidden_activation)
        self.o_act = get_activation(output_activation)

        #loss por nombre
        self.loss_fn = get_loss(loss)

        #pesos 
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.loss_history_ = []

        self._init_params()

    def _init_params(self):
        rng = np.random.default_rng(self.seed)

        #pesos primera capa
        self.W1 = rng.normal(0, 0.1, size=(self.n_inputs, self.n_hidden))
        self.b1 = np.zeros((1, self.n_hidden))

        #pesos salida
        self.W2 = rng.normal(0, 0.1, size=(self.n_hidden, self.n_outputs))
        self.b2 = np.zeros((1, self.n_outputs))

    def forward(self, X):
        X = np.array(X, dtype=float)

        #capa oculta
        Z1 = X @ self.W1 + self.b1
        A1 = self.h_act.forward(Z1)

        #capa de salida
        Z2 = A1 @ self.W2 + self.b2
        A2 = self.o_act.forward(Z2)

        # guardamos cosas pa despues "backpropagatio"
        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2,
        }

        return A2, cache

    def backward(self, y_true, cache):
        X = cache["X"]
        Z1 = cache["Z1"]
        A1 = cache["A1"]
        Z2 = cache["Z2"]
        A2 = cache["A2"]

        y_true = np.array(y_true, dtype=float)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        m = X.shape[0] #datosd

        #gradiente del loss respecto a la salida
        dL_dA2 = self.loss_fn.derivative(y_true, A2)

        #pasamos por activacion de salida
        dA2_dZ2 = self.o_act.derivative(Z2)
        dZ2 = dL_dA2 * dA2_dZ2

        #gradientes capa 2
        dW2 = (A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        #error que baja a la capa oculta
        dA1 = dZ2 @ self.W2.T
        dA1_dZ1 = self.h_act.derivative(Z1)
        dZ1 = dA1 * dA1_dZ1

        #gradientes capa 1
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        #update de pesos
        self.W2 = self.W2 - self.lr * dW2
        self.b2 = self.b2 - self.lr * db2

        self.W1 = self.W1 - self.lr * dW1
        self.b1 = self.b1 - self.lr * db1

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.loss_history_ = []

        for epoca in range(self.epochs):

            y_pred, cache = self.forward(X)

            #calculo de loss
            loss_value = self.loss_fn.forward(y, y_pred)
            self.loss_history_.append(float(loss_value))

            self.backward(y, cache)

            if self.verbose and (epoca % max(1, self.epochs // 10) == 0):
                print(f"epoca {epoca+1}/{self.epochs} | loss: {loss_value:.6f}")

        return self

    def predict_proba(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

    def predict(self, X, threshold=0.5):
        y_pred = self.predict_proba(X)

        #clasificacion binaria
        if y_pred.shape[1] == 1:
            return (y_pred >= threshold).astype(int)

        #multiclass
        return np.argmax(y_pred, axis=1)

    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)

        #binario
        if y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.reshape(-1)
        if y.ndim != 1:
            y = y.reshape(-1)

        return float(np.mean(y_pred == y))
