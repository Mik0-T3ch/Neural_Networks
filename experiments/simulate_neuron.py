import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.activation import Step, Sigmoid, ReLU, Tanh


def run_neuron_simulation():
    inputs = np.array([0.7, -0.4, 0.9])
    weights = np.array([0.5, 0.8, -0.3])
    bias = 0.2

    weighted_products = inputs * weights
    z = np.sum(weighted_products) + bias

    step_fn = Step(threshold=0.0)
    sigmoid_fn = Sigmoid()
    relu_fn = ReLU()
    tanh_fn = Tanh()

    out_step = step_fn.forward(z)
    out_sigmoid = sigmoid_fn.forward(z)
    out_relu = relu_fn.forward(z)
    out_tanh = tanh_fn.forward(z)

    print("==================================================")
    print("      SIMULACION DE UNA NEURONA ARTIFICIAL       ")
    print("==================================================")
    print(f"Entradas (Dendritas / x)        : {inputs}")
    print(f"Pesos (Sinapsis / w)            : {weights}")
    print(f"Sesgo (Bias / b)                : {bias}")
    print(f"Productos ponderados (w * x)    : {weighted_products}")
    print(f"Suma de integracion (Soma / z)  : {z:.4f}")
    print("--------------------------------------------------")
    print("Respuestas de Activacion (Potencial de Accion / Axon):")
    print(f" - Heaviside Step : {out_step}")
    print(f" - Sigmoide       : {out_sigmoid:.4f}")
    print(f" - ReLU           : {out_relu:.4f}")
    print(f" - Tanh           : {out_tanh:.4f}")
    print("==================================================")

    z_sweep = np.linspace(-6, 6, 300)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(z_sweep, step_fn.forward(z_sweep), label="Step (Escalón)", lw=2, linestyle="--")
    ax.plot(z_sweep, sigmoid_fn.forward(z_sweep), label="Sigmoid", lw=2)
    ax.plot(z_sweep, relu_fn.forward(z_sweep), label="ReLU", lw=2)
    ax.plot(z_sweep, tanh_fn.forward(z_sweep), label="Tanh", lw=2)
    ax.axvline(x=z, color="black", linestyle=":", label=f"Soma actual (z={z:.2f})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_title("Simulacion de Activacion Neuronal", fontsize=14, fontweight="bold")
    ax.set_xlabel("Potencial Integrado (z = w · x + b)", fontsize=11)
    ax.set_ylabel("Respuesta Neuronal a(z)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    os.makedirs("assets", exist_ok=True)
    save_path = os.path.join("assets", "neuron_simulation.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Grafico guardado en: {save_path}")


if __name__ == "__main__":
    run_neuron_simulation()
