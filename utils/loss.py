import numpy as np


class Loss:
    def __init__(self, name="base"):
        self.name = name

    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MSE(Loss):
    def __init__(self):
        super().__init__("mse")

    def forward(self, y_true, y_pred):
        diff = y_true - y_pred
        return float(np.mean(diff ** 2))

    def derivative(self, y_true, y_pred):
        return 2.0 * (y_pred - y_true)


class MAE(Loss):
    def __init__(self):
        super().__init__("mae")

    def forward(self, y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    def derivative(self, y_true, y_pred):
        return np.sign(y_pred - y_true)


class BinaryCrossEntropy(Loss):
    def __init__(self, eps=1e-12):
        super().__init__("bce")
        self.eps = eps

    def forward(self, y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, self.eps, 1.0 - self.eps)
        term1 = y_true * np.log(y_pred_clipped)
        term2 = (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        return float(-np.mean(term1 + term2))

    def derivative(self, y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, self.eps, 1.0 - self.eps)
        num = y_pred_clipped - y_true
        den = y_pred_clipped * (1.0 - y_pred_clipped)
        return num / den


class CategoricalCrossEntropy(Loss):
    def __init__(self, eps=1e-12):
        super().__init__("cce")
        self.eps = eps

    def forward(self, y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, self.eps, 1.0 - self.eps)
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.eye(n_classes)[y_true.astype(int)]
        else:
            y_one_hot = y_true
        return float(-np.mean(np.sum(y_one_hot * np.log(y_pred_clipped), axis=-1)))

    def derivative(self, y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, self.eps, 1.0 - self.eps)
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.eye(n_classes)[y_true.astype(int)]
        else:
            y_one_hot = y_true
        return - (y_one_hot / y_pred_clipped)


LOSSES = {
    "mse": MSE(),
    "mae": MAE(),
    "bce": BinaryCrossEntropy(),
    "binary_crossentropy": BinaryCrossEntropy(),
    "cce": CategoricalCrossEntropy(),
    "categorical_crossentropy": CategoricalCrossEntropy(),
    "cross_entropy": CategoricalCrossEntropy(),
}


def get_loss(nombre):
    if isinstance(nombre, Loss):
        return nombre
    nombre_key = str(nombre).lower().strip()
    if nombre_key not in LOSSES:
        raise ValueError(f"Funcion de perdida '{nombre}' no soportada. Opciones: {list(LOSSES.keys())}")
    return LOSSES[nombre_key]
