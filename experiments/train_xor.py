import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.perceptron import Perceptron
from models.mlp import MLP
from utils.datasets import make_xor
from utils.visualization import plot_decision_boundary, plot_loss_history


def run_xor_comparison():
    X, y = make_xor()
    os.makedirs("assets", exist_ok=True)

    print("==================================================")
    print("        EXPERIMENTO XOR: PERCEPTRON VS MLP        ")
    print("==================================================")
    print("Dataset XOR:")
    for xi, yi in zip(X, y):
        print(f"  Entrada: {xi.astype(int)} -> Salida esperada: {int(yi)}")

    print("\n--- 1. Entrenando Perceptron Simple ---")
    perceptron = Perceptron(lr=0.1, epochs=100, seed=42)
    perceptron.fit(X, y)
    p_acc = perceptron.score(X, y)
    print(f"Perceptron Accuracy: {p_acc * 100:.1f}%")
    print(f"Perceptron Predicciones: {perceptron.predict(X).tolist()}")

    save_p = os.path.join("assets", "perceptron_xor_fail.png")
    plot_decision_boundary(perceptron, X, y, title="Perceptron Monocapa en XOR (Falla)", save_path=save_p)

    print("\n--- 2. Entrenando MLP (Red Neuronal Multicapa 2 -> 4 -> 1) ---")
    mlp = MLP(
        n_inputs=2,
        hidden_layers=[4],
        n_outputs=1,
        lr=0.5,
        epochs=3000,
        hidden_activation="tanh",
        output_activation="sigmoid",
        loss="bce",
        seed=42,
        verbose=False,
    )
    mlp.fit(X, y)
    mlp_acc = mlp.score(X, y)
    mlp_probs = mlp.predict_proba(X).reshape(-1)
    mlp_preds = mlp.predict(X).tolist()

    print(f"MLP Accuracy           : {mlp_acc * 100:.1f}%")
    print(f"MLP Probabilidades     : {[round(float(p), 4) for p in mlp_probs]}")
    print(f"MLP Predicciones Finales: {mlp_preds}")

    save_mlp = os.path.join("assets", "mlp_xor_success.png")
    plot_decision_boundary(mlp, X, y, title="MLP en XOR (Separacion No Lineal Exitosa)", save_path=save_mlp)

    save_loss = os.path.join("assets", "mlp_xor_loss.png")
    plot_loss_history(mlp.loss_history_, title="Convergencia de Perdida (Loss) MLP en XOR", save_path=save_loss)

    print("\n==================================================")
    print(f"Graficos guardados en carpeta 'assets/'")
    print("==================================================")


if __name__ == "__main__":
    run_xor_comparison()
