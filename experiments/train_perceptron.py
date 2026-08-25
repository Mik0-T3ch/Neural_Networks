import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.perceptron import Perceptron
from utils.datasets import make_and, make_or, make_nand, make_linearly_separable
from utils.visualization import plot_decision_boundary


def train_perceptron_gates():
    gates = [
        ("AND", make_and),
        ("OR", make_or),
        ("NAND", make_nand),
    ]

    os.makedirs("assets", exist_ok=True)

    print("==================================================")
    print("      ENTRENAMIENTO DE PERCEPTRON MONOCAPA        ")
    print("==================================================")

    for name, dataset_fn in gates:
        X, y = dataset_fn()
        model = Perceptron(lr=0.1, epochs=20, seed=42)
        model.fit(X, y)

        acc = model.score(X, y)
        w_bias = model.w[0]
        w1 = model.w[1]
        w2 = model.w[2]

        print(f"\n--- Compuerta {name} ---")
        print(f"Pesos finales : w_bias={w_bias:.4f}, w1={w1:.4f}, w2={w2:.4f}")
        print(f"Epocas hasta converger : {len(model.errors_)}")
        print(f"Historial de errores   : {model.errors_}")
        print(f"Precision (Accuracy)   : {acc * 100:.1f}%")

        for xi, yi in zip(X, y):
            pred = model.predict(xi)[0]
            print(f" Entrada: {xi.astype(int)} -> Esperado: {int(yi)} | Predicho: {pred}")

        save_path = os.path.join("assets", f"perceptron_{name.lower()}.png")
        plot_decision_boundary(model, X, y, title=f"Perceptron - Compuerta {name}", save_path=save_path)

    print("\n--- Dataset Linealmente Separable (100 muestras) ---")
    X_lin, y_lin = make_linearly_separable(n_samples=100, seed=42)
    model_lin = Perceptron(lr=0.05, epochs=50, seed=42)
    model_lin.fit(X_lin, y_lin)
    acc_lin = model_lin.score(X_lin, y_lin)

    print(f"Precision en dataset sintético: {acc_lin * 100:.1f}%")
    save_path_lin = os.path.join("assets", "perceptron_linearly_separable.png")
    plot_decision_boundary(model_lin, X_lin, y_lin, title="Perceptron - Separacion Lineal", save_path=save_path_lin)
    print(f"Graficos guardados en carpeta 'assets/'")
    print("==================================================")


if __name__ == "__main__":
    train_perceptron_gates()
