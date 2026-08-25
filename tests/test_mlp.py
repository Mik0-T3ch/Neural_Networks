import numpy as np
from models.mlp import MLP
from utils.datasets import make_xor, make_circles, make_spiral


def test_mlp_xor_convergence():
    X, y = make_xor()
    mlp = MLP(
        n_inputs=2,
        hidden_layers=[4],
        n_outputs=1,
        lr=0.5,
        epochs=2500,
        seed=42,
    )
    mlp.fit(X, y)
    assert mlp.score(X, y) == 1.0


def test_mlp_circles_convergence():
    X, y = make_circles(n_samples=100, noise=0.05, factor=0.4, seed=42)
    mlp = MLP(
        n_inputs=2,
        hidden_layers=[12, 6],
        n_outputs=1,
        lr=0.2,
        epochs=1000,
        seed=42,
    )
    mlp.fit(X, y)
    assert mlp.score(X, y) >= 0.95


def test_mlp_multiclass_spiral():
    X, y = make_spiral(n_samples_per_class=40, n_classes=3, noise=0.1, seed=42)
    mlp = MLP(
        n_inputs=2,
        hidden_layers=[24, 12],
        n_outputs=3,
        lr=0.1,
        epochs=1500,
        hidden_activation="relu",
        output_activation="softmax",
        loss="cce",
        seed=42,
    )
    mlp.fit(X, y)
    assert mlp.score(X, y) >= 0.90
