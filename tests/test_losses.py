import numpy as np
import pytest
from utils.loss import (
    MSE,
    MAE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    get_loss,
)


def test_mse():
    loss = MSE()
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([1.0, 0.0])
    assert np.isclose(loss.forward(y_true, y_pred), 0.0)

    y_pred_bad = np.array([0.0, 1.0])
    assert np.isclose(loss.forward(y_true, y_pred_bad), 1.0)
    der = loss.derivative(y_true, y_pred_bad)
    assert np.array_equal(der, np.array([-2.0, 2.0]))


def test_mae():
    loss = MAE()
    y_true = np.array([2.0, 3.0])
    y_pred = np.array([1.0, 5.0])
    assert np.isclose(loss.forward(y_true, y_pred), 1.5)


def test_binary_cross_entropy():
    loss = BinaryCrossEntropy()
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.999999, 0.000001])
    val = loss.forward(y_true, y_pred)
    assert val < 0.01

    der = loss.derivative(y_true, y_pred)
    assert der.shape == y_true.shape


def test_categorical_cross_entropy():
    loss = CategoricalCrossEntropy()
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])
    val = loss.forward(y_true, y_pred)
    assert val < 0.2


def test_get_loss_invalid():
    with pytest.raises(ValueError):
        get_loss("perdida_desconocida")
