import numpy as np
import pytest
from utils.activation import (
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Linear,
    GELU,
    Softmax,
    Step,
    get_activation,
)


def test_sigmoid():
    act = Sigmoid()
    x = np.array([0.0, 2.0, -2.0])
    out = act.forward(x)
    assert np.isclose(out[0], 0.5)
    assert np.all(out > 0.0) and np.all(out < 1.0)
    der = act.derivative(x)
    assert np.isclose(der[0], 0.25)


def test_relu():
    act = ReLU()
    x = np.array([-3.0, 0.0, 3.0])
    out = act.forward(x)
    assert np.array_equal(out, np.array([0.0, 0.0, 3.0]))
    der = act.derivative(x)
    assert np.array_equal(der, np.array([0.0, 0.0, 1.0]))


def test_leaky_relu():
    act = LeakyReLU(alpha=0.1)
    x = np.array([-2.0, 2.0])
    out = act.forward(x)
    assert np.isclose(out[0], -0.2)
    assert np.isclose(out[1], 2.0)
    der = act.derivative(x)
    assert np.isclose(der[0], 0.1)
    assert np.isclose(der[1], 1.0)


def test_tanh():
    act = Tanh()
    x = np.array([0.0])
    assert np.isclose(act.forward(x)[0], 0.0)
    assert np.isclose(act.derivative(x)[0], 1.0)


def test_linear():
    act = Linear()
    x = np.array([1.5, -3.2])
    assert np.array_equal(act.forward(x), x)
    assert np.array_equal(act.derivative(x), np.ones_like(x))


def test_gelu():
    act = GELU()
    x = np.array([0.0, 1.0])
    out = act.forward(x)
    assert np.isclose(out[0], 0.0)
    assert out[1] > 0.8
    der = act.derivative(x)
    assert der.shape == x.shape


def test_softmax():
    act = Softmax()
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = act.forward(x)
    assert np.allclose(np.sum(out, axis=1), np.array([1.0, 1.0]))
    assert np.all(out >= 0.0)


def test_step():
    act = Step(threshold=0.0)
    x = np.array([-1.0, 0.0, 1.0])
    assert np.array_equal(act.forward(x), np.array([0.0, 1.0, 1.0]))


def test_get_activation_invalid():
    with pytest.raises(ValueError):
        get_activation("inexistente")
