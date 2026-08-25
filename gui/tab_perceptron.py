import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from models.perceptron import Perceptron
from utils.datasets import make_and, make_or, make_nand, make_linearly_separable


class PerceptronTrainerTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.custom_points_X = []
        self.custom_points_y = []
        self._build_ui()
        self._load_dataset()

    def _build_ui(self):
        control_frame = ttk.LabelFrame(self, text="Configuracion del Perceptron", padding=12)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(control_frame, text="Dataset de Entrada:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.dataset_var = tk.StringVar(value="AND")
        dataset_combo = ttk.Combobox(
            control_frame,
            textvariable=self.dataset_var,
            values=["AND", "OR", "NAND", "Linealmente Separable", "Personalizado (Clics)"],
            state="readonly"
        )
        dataset_combo.pack(fill=tk.X, pady=(0, 10))
        dataset_combo.bind("<<ComboboxSelected>>", lambda e: self._load_dataset())

        ttk.Label(control_frame, text="Tasa de Aprendizaje (lr):", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Scale(control_frame, from_=0.01, to=1.0, variable=self.lr_var).pack(fill=tk.X, pady=(0, 10))

        ttk.Label(control_frame, text="Epocas Maximas:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.epochs_var = tk.IntVar(value=30)
        ttk.Scale(control_frame, from_=5, to=100, variable=self.epochs_var).pack(fill=tk.X, pady=(0, 15))

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Entrenar Perceptron", command=self._train_perceptron).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="Limpiar Clics", command=self._clear_custom_points).pack(fill=tk.X, pady=3)

        self.status_label = ttk.Label(
            control_frame,
            text="Selecciona dataset y presiona 'Entrenar'",
            font=("Consolas", 9),
            background="#f8f9fa",
            padding=8,
            relief="groove",
            justify=tk.LEFT,
            wraplength=200
        )
        self.status_label.pack(fill=tk.X, pady=10)

        help_box = ttk.Label(
            control_frame,
            text="Tip Clics:\nClic Izquierdo = Clase 0 (Azul)\nClic Derecho = Clase 1 (Rojo)",
            font=("Segoe UI", 8),
            foreground="#555"
        )
        help_box.pack(anchor=tk.W, pady=5)

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(7.5, 5), dpi=100)
        self.ax_plane = self.fig.add_subplot(121)
        self.ax_errors = self.fig.add_subplot(122)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def _load_dataset(self):
        ds_name = self.dataset_var.get()
        if ds_name == "AND":
            self.X, self.y = make_and()
        elif ds_name == "OR":
            self.X, self.y = make_or()
        elif ds_name == "NAND":
            self.X, self.y = make_nand()
        elif ds_name == "Linealmente Separable":
            self.X, self.y = make_linearly_separable(n_samples=50, seed=42)
        else:
            self._render_custom_points()
            return

        self._render_data_only()

    def _on_canvas_click(self, event):
        if self.dataset_var.get() != "Personalizado (Clics)":
            return
        if event.inaxes != self.ax_plane:
            return
        if event.xdata is None or event.ydata is None:
            return

        label = 0 if event.button == 1 else 1
        self.custom_points_X.append([event.xdata, event.ydata])
        self.custom_points_y.append(label)
        self._render_custom_points()

    def _clear_custom_points(self):
        self.custom_points_X = []
        self.custom_points_y = []
        self._render_custom_points()

    def _render_custom_points(self):
        if len(self.custom_points_X) > 0:
            self.X = np.array(self.custom_points_X, dtype=float)
            self.y = np.array(self.custom_points_y, dtype=float)
        else:
            self.X = np.empty((0, 2))
            self.y = np.empty((0,))

        self._render_data_only()

    def _render_data_only(self):
        self.ax_plane.clear()
        self.ax_errors.clear()

        if len(self.X) > 0:
            colors = ["#1f77b4" if yi == 0 else "#d62728" for yi in self.y]
            self.ax_plane.scatter(self.X[:, 0], self.X[:, 1], c=colors, s=70, edgecolors="k", zorder=5)
            x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
            y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
            self.ax_plane.set_xlim(x_min, x_max)
            self.ax_plane.set_ylim(y_min, y_max)
        else:
            self.ax_plane.set_xlim(-2, 2)
            self.ax_plane.set_ylim(-2, 2)

        self.ax_plane.set_title("Plano 2D (Entradas y Clases)", fontsize=11, fontweight="bold")
        self.ax_plane.set_xlabel("x1")
        self.ax_plane.set_ylabel("x2")
        self.ax_plane.grid(True, linestyle=":", alpha=0.6)

        self.ax_errors.set_title("Evolucion de Errores", fontsize=11, fontweight="bold")
        self.ax_errors.set_xlabel("Epoca")
        self.ax_errors.set_ylabel("Errores")
        self.ax_errors.grid(True, linestyle=":", alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw()

    def _train_perceptron(self):
        if len(self.X) < 2 or len(np.unique(self.y)) < 2:
            self.status_label.config(text="Se necesitan al menos 2 puntos de clases distintas.")
            return

        lr = self.lr_var.get()
        epochs = self.epochs_var.get()

        model = Perceptron(lr=lr, epochs=epochs, seed=42)
        model.fit(self.X, self.y)

        acc = model.score(self.X, self.y)
        w_b, w1, w2 = model.w[0], model.w[1], model.w[2]

        info = (
            f"Precision: {acc * 100:.1f}%\n"
            f"Epocas: {len(model.errors_)}\n"
            f"w_bias: {w_b:+.4f}\n"
            f"w1: {w1:+.4f} | w2: {w2:+.4f}"
        )
        self.status_label.config(text=info)

        self.ax_plane.clear()
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = model.predict(grid).reshape(xx.shape)

        self.ax_plane.contourf(xx, yy, preds, alpha=0.3, cmap="coolwarm")
        colors = ["#1f77b4" if yi == 0 else "#d62728" for yi in self.y]
        self.ax_plane.scatter(self.X[:, 0], self.X[:, 1], c=colors, s=70, edgecolors="k", zorder=5)

        if abs(w2) > 1e-6:
            line_x = np.array([x_min, x_max])
            line_y = -(w1 * line_x + w_b) / w2
            self.ax_plane.plot(line_x, line_y, color="black", lw=2, linestyle="--", label="Recta Separadora")
            self.ax_plane.legend(loc="upper right", fontsize=8)

        self.ax_plane.set_xlim(x_min, x_max)
        self.ax_plane.set_ylim(y_min, y_max)
        self.ax_plane.set_title("Frontera Lineal del Perceptron", fontsize=11, fontweight="bold")
        self.ax_plane.set_xlabel("x1")
        self.ax_plane.set_ylabel("x2")
        self.ax_plane.grid(True, linestyle=":", alpha=0.6)

        self.ax_errors.clear()
        self.ax_errors.plot(range(1, len(model.errors_) + 1), model.errors_, marker="o", color="#d62728", lw=2)
        self.ax_errors.set_title("Errores por Epoca", fontsize=11, fontweight="bold")
        self.ax_errors.set_xlabel("Epoca")
        self.ax_errors.set_ylabel("Cantidad de Errores")
        self.ax_errors.grid(True, linestyle=":", alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw()
