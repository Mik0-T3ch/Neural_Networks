import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import MLP
from utils.datasets import make_circles, make_moons, make_spiral
from utils.visualization import plot_decision_boundary, plot_loss_history


def run_mlp_experiments():
    os.makedirs("assets", exist_ok=True)

    print("==================================================")
    print("      ENTRENAMIENTO DE MLP EN PATRONES COMPLEJOS  ")
    print("==================================================")

    print("\n--- 1. Clasificacion de Circulos Concentricos ---")
    X_circ, y_circ = make_circles(n_samples=300, noise=0.08, factor=0.4, seed=42)
    mlp_circ = MLP(
        n_inputs=2,
        hidden_layers=[16, 8],
        n_outputs=1,
        lr=0.2,
        epochs=1500,
        batch_size=32,
        hidden_activation="tanh",
        output_activation="sigmoid",
        loss="bce",
        seed=42,
        verbose=False,
    )
    mlp_circ.fit(X_circ, y_circ)
    acc_circ = mlp_circ.score(X_circ, y_circ)
    print(f"Precision en Circulos: {acc_circ * 100:.2f}% | Loss final: {mlp_circ.loss_history_[-1]:.5f}")
    plot_decision_boundary(mlp_circ, X_circ, y_circ, title="MLP - Circulos Concentricos", save_path=os.path.join("assets", "mlp_circles.png"))
    plot_loss_history(mlp_circ.loss_history_, title="Loss - Circulos Concentricos", save_path=os.path.join("assets", "mlp_circles_loss.png"))

    print("\n--- 2. Clasificacion del Dataset Two Moons ---")
    X_moons, y_moons = make_moons(n_samples=300, noise=0.15, seed=42)
    mlp_moons = MLP(
        n_inputs=2,
        hidden_layers=[16, 8],
        n_outputs=1,
        lr=0.1,
        epochs=1200,
        batch_size=32,
        hidden_activation="relu",
        output_activation="sigmoid",
        loss="bce",
        seed=42,
        verbose=False,
    )
    mlp_moons.fit(X_moons, y_moons)
    acc_moons = mlp_moons.score(X_moons, y_moons)
    print(f"Precision en Moons: {acc_moons * 100:.2f}% | Loss final: {mlp_moons.loss_history_[-1]:.5f}")
    plot_decision_boundary(mlp_moons, X_moons, y_moons, title="MLP - Two Moons", save_path=os.path.join("assets", "mlp_moons.png"))
    plot_loss_history(mlp_moons.loss_history_, title="Loss - Two Moons", save_path=os.path.join("assets", "mlp_moons_loss.png"))

    print("\n--- 3. Clasificacion Multiclase (Espiral 3 Clases) ---")
    X_spiral, y_spiral = make_spiral(n_samples_per_class=100, n_classes=3, noise=0.15, seed=42)
    mlp_spiral = MLP(
        n_inputs=2,
        hidden_layers=[32, 16],
        n_outputs=3,
        lr=0.1,
        epochs=2000,
        batch_size=32,
        hidden_activation="relu",
        output_activation="softmax",
        loss="cce",
        seed=42,
        verbose=False,
    )
    mlp_spiral.fit(X_spiral, y_spiral)
    acc_spiral = mlp_spiral.score(X_spiral, y_spiral)
    print(f"Precision en Espiral: {acc_spiral * 100:.2f}% | Loss final: {mlp_spiral.loss_history_[-1]:.5f}")
    plot_decision_boundary(mlp_spiral, X_spiral, y_spiral, title="MLP Multiclase - Espiral 3 Clases", save_path=os.path.join("assets", "mlp_spiral.png"))
    plot_loss_history(mlp_spiral.loss_history_, title="Loss - Espiral 3 Clases", save_path=os.path.join("assets", "mlp_spiral_loss.png"))

    print("\n==================================================")
    print(f"Todos los graficos guardados correctamente en 'assets/'")
    print("==================================================")


if __name__ == "__main__":
    run_mlp_experiments()
