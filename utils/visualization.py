import matplotlib.pyplot as plt
import numpy as np


def plot_decision_boundary(model, X, y, title="Frontera de Decision", save_path=None, show=False):
    X = np.array(X, dtype=float)
    y = np.array(y)
    
    if y.ndim > 1:
        y = y.reshape(-1)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    try:
        preds = model.predict(grid_points)
    except Exception:
        preds = model.forward(grid_points)[0]
        if preds.shape[1] == 1:
            preds = (preds >= 0.5).astype(int)
        else:
            preds = np.argmax(preds, axis=1)
            
    if hasattr(preds, "ndim") and preds.ndim > 1 and preds.shape[1] == 1:
        preds = preds.reshape(-1)
        
    zz = preds.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, zz, alpha=0.35, cmap="coolwarm")
    ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.5, linestyles="--")
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=60, linewidths=1)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Caracteristica 1 (x1)", fontsize=11)
    ax.set_ylabel("Caracteristica 2 (x2)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_loss_history(loss_history, title="Curva de Perdida", save_path=None, show=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, color="#2b5c8f", lw=2, label="Perdida (Loss)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("Valor de Perdida", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_activation_functions(save_path=None, show=False):
    from utils.activation import Sigmoid, ReLU, LeakyReLU, Tanh, GELU

    x = np.linspace(-5, 5, 400)
    acts = [
        ("Sigmoid", Sigmoid()),
        ("Tanh", Tanh()),
        ("ReLU", ReLU()),
        ("LeakyReLU", LeakyReLU(0.1)),
        ("GELU", GELU()),
    ]
    
    fig, axes = plt.subplots(len(acts), 2, figsize=(10, 12))
    
    for i, (name, act) in enumerate(acts):
        y_fwd = act.forward(x)
        y_der = act.derivative(x)
        
        axes[i, 0].plot(x, y_fwd, color="#1f77b4", lw=2)
        axes[i, 0].set_title(f"{name} - Forward", fontsize=11, fontweight="bold")
        axes[i, 0].grid(True, linestyle=":", alpha=0.6)
        
        axes[i, 1].plot(x, y_der, color="#d62728", lw=2)
        axes[i, 1].set_title(f"{name} - Derivative", fontsize=11, fontweight="bold")
        axes[i, 1].grid(True, linestyle=":", alpha=0.6)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
