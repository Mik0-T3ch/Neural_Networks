import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from utils.activation import get_activation


class NeuronSimulatorTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self._build_ui()
        self._update_simulation()

    def _build_ui(self):
        control_frame = ttk.LabelFrame(self, text="Parametros de la Neurona", padding=12)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.x1_var = tk.DoubleVar(value=0.6)
        self.x2_var = tk.DoubleVar(value=-0.4)
        self.w1_var = tk.DoubleVar(value=0.8)
        self.w2_var = tk.DoubleVar(value=-0.5)
        self.bias_var = tk.DoubleVar(value=0.2)
        self.act_var = tk.StringVar(value="sigmoid")

        self._create_slider(control_frame, "Entrada x1 (Dendrita 1):", self.x1_var, -3.0, 3.0)
        self._create_slider(control_frame, "Entrada x2 (Dendrita 2):", self.x2_var, -3.0, 3.0)
        self._create_slider(control_frame, "Peso w1 (Sinapsis 1):", self.w1_var, -3.0, 3.0)
        self._create_slider(control_frame, "Peso w2 (Sinapsis 2):", self.w2_var, -3.0, 3.0)
        self._create_slider(control_frame, "Sesgo b (Bias):", self.bias_var, -3.0, 3.0)

        ttk.Label(control_frame, text="Funcion de Activacion:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(10, 2))
        act_combo = ttk.Combobox(
            control_frame,
            textvariable=self.act_var,
            values=["sigmoid", "relu", "leaky_relu", "tanh", "gelu", "step", "linear"],
            state="readonly"
        )
        act_combo.pack(fill=tk.X, pady=(0, 15))
        act_combo.bind("<<ComboboxSelected>>", lambda e: self._update_simulation())

        self.readout_label = ttk.Label(
            control_frame,
            text="",
            font=("Consolas", 10),
            background="#f0f4f8",
            padding=10,
            relief="groove",
            justify=tk.LEFT
        )
        self.readout_label.pack(fill=tk.X, pady=10)

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax_curve = self.fig.add_subplot(121)
        self.ax_diagram = self.fig.add_subplot(122)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_slider(self, parent, text, var, min_val, max_val):
        ttk.Label(parent, text=text, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(6, 0))
        scale = ttk.Scale(
            parent,
            from_=min_val,
            to=max_val,
            variable=var,
            command=lambda v: self._update_simulation()
        )
        scale.pack(fill=tk.X, pady=(0, 4))

    def _update_simulation(self):
        x1 = self.x1_var.get()
        x2 = self.x2_var.get()
        w1 = self.w1_var.get()
        w2 = self.w2_var.get()
        b = self.bias_var.get()
        act_name = self.act_var.get()

        z = (x1 * w1) + (x2 * w2) + b
        act_fn = get_activation(act_name)
        out = float(act_fn.forward(np.array([z]))[0])

        info_text = (
            f"x1 = {x1:+.2f}  |  w1 = {w1:+.2f}\n"
            f"x2 = {x2:+.2f}  |  w2 = {w2:+.2f}\n"
            f"Sesgo (b) = {b:+.2f}\n"
            f"---------------------------\n"
            f"Soma z    = {z:+.4f}\n"
            f"Salida a  = {out:+.4f}"
        )
        self.readout_label.config(text=info_text)

        self.ax_curve.clear()
        z_range = np.linspace(-6, 6, 200)
        y_vals = act_fn.forward(z_range)
        self.ax_curve.plot(z_range, y_vals, color="#1f77b4", lw=2, label=f"Activacion ({act_name})")
        self.ax_curve.axvline(z, color="#d62728", linestyle="--", lw=1.5, label=f"z = {z:.2f}")
        self.ax_curve.scatter([z], [out], color="#d62728", s=80, zorder=5)
        self.ax_curve.set_title("Curva de Activacion", fontsize=11, fontweight="bold")
        self.ax_curve.set_xlabel("Soma (z)")
        self.ax_curve.set_ylabel("Salida a(z)")
        self.ax_curve.grid(True, linestyle=":", alpha=0.6)
        self.ax_curve.legend(fontsize=8, loc="upper left")

        self.ax_diagram.clear()
        self.ax_diagram.set_xlim(-1, 5)
        self.ax_diagram.set_ylim(-1, 5)
        self.ax_diagram.axis("off")

        self.ax_diagram.scatter([0, 0], [3.5, 0.5], color="#2b5c8f", s=400, zorder=4)
        self.ax_diagram.text(0, 3.5, f"x1\n{x1:.2f}", color="white", ha="center", va="center", fontsize=8, fontweight="bold")
        self.ax_diagram.text(0, 0.5, f"x2\n{x2:.2f}", color="white", ha="center", va="center", fontsize=8, fontweight="bold")

        circle_soma = FigureCanvasTkAgg
        self.ax_diagram.scatter([2], [2], color="#ff7f0e", s=1400, zorder=4)
        self.ax_diagram.text(2, 2, f"Σ + b\n{z:.2f}", color="white", ha="center", va="center", fontsize=9, fontweight="bold")

        self.ax_diagram.scatter([4], [2], color="#2ca02c", s=700, zorder=4)
        self.ax_diagram.text(4, 2, f"a(z)\n{out:.2f}", color="white", ha="center", va="center", fontsize=8, fontweight="bold")

        self.ax_diagram.annotate("", xy=(2, 2), xytext=(0, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
        self.ax_diagram.annotate("", xy=(2, 2), xytext=(0, 0.5), arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
        self.ax_diagram.annotate("", xy=(4, 2), xytext=(2, 2), arrowprops=dict(arrowstyle="->", lw=2.5, color="#555"))

        self.ax_diagram.text(0.9, 3.0, f"w1={w1:.2f}", fontsize=8, color="#111", fontweight="bold")
        self.ax_diagram.text(0.9, 1.0, f"w2={w2:.2f}", fontsize=8, color="#111", fontweight="bold")
        self.ax_diagram.text(2, 0.8, f"Bias={b:.2f}", fontsize=9, color="#d62728", ha="center", fontweight="bold")
        self.ax_diagram.set_title("Diagrama de la Neurona", fontsize=11, fontweight="bold")

        self.fig.tight_layout()
        self.canvas.draw()
