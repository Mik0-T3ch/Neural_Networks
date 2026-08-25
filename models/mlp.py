import numpy as np
from utils.activation import get_activation, Sigmoid, Softmax
from utils.loss import get_loss, BinaryCrossEntropy, CategoricalCrossEntropy


class MLP:
    def __init__(
        self,
        n_inputs,
        hidden_layers=(8,),
        n_outputs=1,
        lr=0.1,
        epochs=1000,
        batch_size=None,
        hidden_activation="tanh",
        output_activation="sigmoid",
        loss="bce",
        seed=None,
        verbose=False,
    ):
        self.n_inputs = int(n_inputs)
        if isinstance(hidden_layers, int):
            self.hidden_layers = [hidden_layers]
        else:
            self.hidden_layers = list(hidden_layers)
        self.n_outputs = int(n_outputs)

        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = bool(verbose)

        self.h_act = get_activation(hidden_activation)
        self.o_act = get_activation(output_activation)
        self.loss_fn = get_loss(loss)

        self.weights = []
        self.biases = []
        self.loss_history_ = []

        self._init_parameters()

    def _init_parameters(self):
        rng = np.random.default_rng(self.seed)
        layer_dims = [self.n_inputs] + self.hidden_layers + [self.n_outputs]

        self.weights = []
        self.biases = []

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = rng.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out), dtype=float)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        activations = [X]
        linear_inputs = []

        current_a = X
        n_layers = len(self.weights)

        for i in range(n_layers):
            z = np.dot(current_a, self.weights[i]) + self.biases[i]
            linear_inputs.append(z)
            if i == n_layers - 1:
                current_a = self.o_act.forward(z)
            else:
                current_a = self.h_act.forward(z)
            activations.append(current_a)

        return current_a, {"activations": activations, "linear_inputs": linear_inputs}

    def backward(self, y_true, cache):
        activations = cache["activations"]
        linear_inputs = cache["linear_inputs"]
        n_layers = len(self.weights)
        m = activations[0].shape[0]

        a_last = activations[-1]
        z_last = linear_inputs[-1]

        y_true = np.array(y_true, dtype=float)
        if y_true.ndim == 1:
            if self.n_outputs == 1:
                y_true = y_true.reshape(-1, 1)
            else:
                y_true = np.eye(self.n_outputs)[y_true.astype(int)]

        if isinstance(self.loss_fn, BinaryCrossEntropy) and isinstance(self.o_act, Sigmoid):
            dZ = a_last - y_true
        elif isinstance(self.loss_fn, CategoricalCrossEntropy) and isinstance(self.o_act, Softmax):
            dZ = a_last - y_true
        else:
            dL_dA = self.loss_fn.derivative(y_true, a_last)
            dA_dZ = self.o_act.derivative(z_last)
            dZ = dL_dA * dA_dZ

        grad_w = [None] * n_layers
        grad_b = [None] * n_layers

        for i in reversed(range(n_layers)):
            a_prev = activations[i]
            grad_w[i] = np.dot(a_prev.T, dZ) / m
            grad_b[i] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                z_prev = linear_inputs[i - 1]
                dZ = dA_prev * self.h_act.derivative(z_prev)

        for i in range(n_layers):
            self.weights[i] -= self.lr * grad_w[i]
            self.biases[i] -= self.lr * grad_b[i]

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        if y.ndim == 1 and self.n_outputs == 1:
            y = y.reshape(-1, 1)

        m = X.shape[0]
        rng = np.random.default_rng(self.seed)
        self.loss_history_ = []

        batch_size = self.batch_size if self.batch_size is not None else m

        for epoch in range(self.epochs):
            indices = rng.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, m, batch_size):
                end_idx = min(start_idx + batch_size, m)
                xb = X_shuffled[start_idx:end_idx]
                yb = y_shuffled[start_idx:end_idx]

                _, cache = self.forward(xb)
                self.backward(yb, cache)

            y_pred_full, _ = self.forward(X)
            epoch_loss = self.loss_fn.forward(y, y_pred_full)
            self.loss_history_.append(float(epoch_loss))

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                print(f"Epoca {epoch + 1}/{self.epochs} | Loss: {epoch_loss:.6f}")

        return self

    def predict_proba(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

    def predict(self, X, threshold=0.5):
        y_pred = self.predict_proba(X)
        if self.n_outputs == 1:
            return (y_pred >= threshold).astype(int).reshape(-1)
        return np.argmax(y_pred, axis=-1)

    def score(self, X, y):
        y = np.array(y)
        if y.ndim != 1:
            y = y.reshape(-1)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
