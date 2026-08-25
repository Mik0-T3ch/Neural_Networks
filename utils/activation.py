import numpy as np


class Activation:
    def __init__(self, name="base"):
        self.name = name

    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(Activation):
    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, x):
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    def derivative(self, x):
        s = self.forward(x)
        return s * (1.0 - s)


class ReLU(Activation):
    def __init__(self):
        super().__init__("relu")

    def forward(self, x):
        return np.maximum(0.0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        super().__init__("leaky_relu")
        self.alpha = float(alpha)

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x):
        return np.where(x > 0, 1.0, self.alpha)


class Tanh(Activation):
    def __init__(self):
        super().__init__("tanh")

    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        t = np.tanh(x)
        return 1.0 - t ** 2


class Linear(Activation):
    def __init__(self):
        super().__init__("linear")

    def forward(self, x):
        return np.array(x, dtype=float)

    def derivative(self, x):
        return np.ones_like(x, dtype=float)


class GELU(Activation):
    def __init__(self):
        super().__init__("gelu")

    def forward(self, x):
        c = np.sqrt(2.0 / np.pi)
        x3 = x ** 3
        inside = x + 0.044715 * x3
        tanh_part = np.tanh(c * inside)
        return 0.5 * x * (1.0 + tanh_part)

    def derivative(self, x):
        c = np.sqrt(2.0 / np.pi)
        x2 = x ** 2
        x3 = x ** 3
        inside = x + 0.044715 * x3
        tanh_part = np.tanh(c * inside)
        sech2 = 1.0 - tanh_part ** 2
        parte1 = 0.5 * (1.0 + tanh_part)
        deriv_inside = c * (1.0 + 3.0 * 0.044715 * x2)
        parte2 = 0.5 * x * sech2 * deriv_inside
        return parte1 + parte2


class Softmax(Activation):
    def __init__(self):
        super().__init__("softmax")

    def forward(self, x):
        if x.ndim == 1:
            shifted = x - np.max(x)
            exp_vals = np.exp(shifted)
            return exp_vals / np.sum(exp_vals)
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def derivative(self, x):
        s = self.forward(x)
        return s * (1.0 - s)


class Step(Activation):
    def __init__(self, threshold=0.0):
        super().__init__("step")
        self.threshold = threshold

    def forward(self, x):
        return np.where(x >= self.threshold, 1.0, 0.0)

    def derivative(self, x):
        return np.zeros_like(x, dtype=float)


ACTIVATIONS = {
    "sigmoid": Sigmoid(),
    "relu": ReLU(),
    "leaky_relu": LeakyReLU(),
    "tanh": Tanh(),
    "linear": Linear(),
    "lin": Linear(),
    "gelu": GELU(),
    "softmax": Softmax(),
    "step": Step(),
}


def get_activation(nombre):
    if isinstance(nombre, Activation):
        return nombre
    nombre_key = str(nombre).lower().strip()
    if nombre_key not in ACTIVATIONS:
        raise ValueError(f"Funcion de activacion '{nombre}' no soportada. Opciones: {list(ACTIVATIONS.keys())}")
    return ACTIVATIONS[nombre_key]
