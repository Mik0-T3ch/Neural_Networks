import numpy as np
from models.perceptron import Perceptron
from utils.datasets import make_and, make_or, make_nand, make_linearly_separable


def test_perceptron_and():
    X, y = make_and()
    p = Perceptron(lr=0.1, epochs=50, seed=42)
    p.fit(X, y)
    assert p.score(X, y) == 1.0


def test_perceptron_or():
    X, y = make_or()
    p = Perceptron(lr=0.1, epochs=50, seed=42)
    p.fit(X, y)
    assert p.score(X, y) == 1.0


def test_perceptron_nand():
    X, y = make_nand()
    p = Perceptron(lr=0.1, epochs=50, seed=42)
    p.fit(X, y)
    assert p.score(X, y) == 1.0


def test_perceptron_linearly_separable():
    X, y = make_linearly_separable(n_samples=50, seed=42)
    p = Perceptron(lr=0.05, epochs=50, seed=42)
    p.fit(X, y)
    assert p.score(X, y) >= 0.95


def test_perceptron_prediction_shape():
    X, y = make_and()
    p = Perceptron().fit(X, y)
    preds = p.predict(X)
    assert preds.shape == (4,)
