import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from models.mlp import MLP
from utils.datasets import make_xor, make_circles, make_moons, make_spiral
from gui.data_loader import load_csv_dataset, generate_sample_csv


class MLPTrainerTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.custom_points_X = []
        self.custom_points_y = []
        self.loaded_csv_path = None
        self._build_ui()
        self._load_dataset()

    def _build_ui(self):
        control_frame = ttk.LabelFrame(self, text="Configuracion de la Red Neuronal (MLP)", padding=12)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(control_frame, text="Dataset de Entrenamiento:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.dataset_var = tk.StringVar(value="XOR")
        dataset_combo = ttk.Combobox(
            control_frame,
            textvariable=self.dataset_var,
            values=[
                "XOR",
                "Circulos Concentricos",
                "Two Moons",
                "Espiral (3 Clases)",
                "Cargar Archivo CSV",
                "Personalizado (Clics)",
            ],
            state="readonly"
        )
        dataset_combo.pack(fill=tk.X, pady=(0, 8))
        dataset_combo.bind("<<ComboboxSelected>>", lambda e: self._on_dataset_change())

        self.csv_btn_frame = ttk.Frame(control_frame)
        ttk.Button(self.csv_btn_frame, text="Seleccionar CSV...", command=self._select_csv_file).pack(fill=tk.X, pady=2)
        ttk.Button(self.csv_btn_frame, text="Generar Ejemplo CSV", command=self._export_sample_csv).pack(fill=tk.X, pady=2)

        ttk.Label(control_frame, text="Capas Ocultas (ej: 16, 8):", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(6, 0))
        self.hidden_layers_var = tk.StringVar(value="16, 8")
        ttk.Entry(control_frame, textvariable=self.hidden_layers_var).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(control_frame, text="Activacion Oculta:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.h_act_var = tk.StringVar(value="tanh")
        ttk.Combobox(
            control_frame,
            textvariable=self.h_act_var,
            values=["tanh", "relu", "leaky_relu", "gelu", "sigmoid"],
            state="readonly"
        ).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(control_frame, text="Tasa de Aprendizaje (lr):", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.lr_var = tk.DoubleVar(value=0.1)
        ttk.Scale(control_frame, from_=0.01, to=1.0, variable=self.lr_var).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(control_frame, text="Epocas:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.epochs_var = tk.IntVar(value=1000)
        ttk.Scale(control_frame, from_=100, to=4000, variable=self.epochs_var).pack(fill=tk.X, pady=(0, 12))

        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Entrenar Red Neuronal", command=self._train_mlp).pack(fill=tk.X, pady=3)
        ttk.Button(btn_frame, text="Limpiar Clics", command=self._clear_custom_points).pack(fill=tk.X, pady=3)

        self.status_label = ttk.Label(
            control_frame,
            text="Listo para entrenar.",
            font=("Consolas", 9),
            background="#f8f9fa",
            padding=8,
            relief="groove",
            justify=tk.LEFT,
            wraplength=210
        )
        self.status_label.pack(fill=tk.X, pady=8)

        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(7.5, 5), dpi=100)
        self.ax_boundary = self.fig.add_subplot(121)
        self.ax_loss = self.fig.add_subplot(122)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def _on_dataset_change(self):
        ds_name = self.dataset_var.get()
        if ds_name == "Cargar Archivo CSV":
            self.csv_btn_frame.pack(fill=tk.X, pady=4)
        else:
            self.csv_btn_frame.pack_forget()

        if ds_name == "Espiral (3 Clases)":
            self.h_act_var.set("relu")
        elif ds_name == "XOR":
            self.h_act_var.set("tanh")
            self.hidden_layers_var.set("4")

        self._load_dataset()

    def _select_csv_file(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if file_path:
            try:
                self.X, self.y = load_csv_dataset(file_path)
                self.loaded_csv_path = file_path
                self.status_label.config(text=f"CSV cargado:\n{len(self.X)} filas, {self.X.shape[1]} caracteristicas.")
                self._render_data_only()
            except Exception as e:
                messagebox.showerror("Error al leer CSV", str(e))

    def _export_sample_csv(self):
        file_path = filedialog.asksaveasfilename(
            title="Guardar archivo CSV de ejemplo",
            defaultextension=".csv",
            filetypes=[("Archivos CSV", "*.csv")]
        )
        if file_path:
            try:
                generate_sample_csv(file_path, dataset_type="moons")
                messagebox.showinfo("Exito", f"Archivo CSV de ejemplo guardado en:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error al guardar CSV", str(e))

    def _load_dataset(self):
        ds_name = self.dataset_var.get()
        if ds_name == "XOR":
            self.X, self.y = make_xor()
        elif ds_name == "Circulos Concentricos":
            self.X, self.y = make_circles(n_samples=250, noise=0.08, factor=0.4, seed=42)
        elif ds_name == "Two Moons":
            self.X, self.y = make_moons(n_samples=250, noise=0.12, seed=42)
        elif ds_name == "Espiral (3 Clases)":
            self.X, self.y = make_spiral(n_samples_per_class=70, n_classes=3, noise=0.15, seed=42)
        elif ds_name == "Cargar Archivo CSV":
            if self.loaded_csv_path is None:
                self.X = np.empty((0, 2))
                self.y = np.empty((0,))
        else:
            self._render_custom_points()
            return

        self._render_data_only()

    def _on_canvas_click(self, event):
        if self.dataset_var.get() != "Personalizado (Clics)":
            return
        if event.inaxes != self.ax_boundary:
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
        self.ax_boundary.clear()
        self.ax_loss.clear()

        if len(self.X) > 0 and self.X.shape[1] == 2:
            classes = np.unique(self.y)
            cmap = "coolwarm" if len(classes) <= 2 else "tab10"
            self.ax_boundary.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=cmap, s=50, edgecolors="k", zorder=5)
            x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
            y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
            self.ax_boundary.set_xlim(x_min, x_max)
            self.ax_boundary.set_ylim(y_min, y_max)
        else:
            self.ax_boundary.set_xlim(-2, 2)
            self.ax_boundary.set_ylim(-2, 2)

        self.ax_boundary.set_title("Frontera No Lineal", fontsize=11, fontweight="bold")
        self.ax_boundary.set_xlabel("x1")
        self.ax_boundary.set_ylabel("x2")
        self.ax_boundary.grid(True, linestyle=":", alpha=0.6)

        self.ax_loss.set_title("Curva de Perdida (Loss)", fontsize=11, fontweight="bold")
        self.ax_loss.set_xlabel("Epoca")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True, linestyle=":", alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw()

    def _train_mlp(self):
        if len(self.X) < 2 or len(np.unique(self.y)) < 2:
            self.status_label.config(text="Se requieren al menos 2 clases distintas.")
            return

        try:
            layers_text = self.hidden_layers_var.get().strip()
            hidden_layers = [int(v.strip()) for v in layers_text.split(",") if v.strip()]
        except Exception:
            self.status_label.config(text="Error: Capas ocultas invalidas. Usa formato '16, 8'")
            return

        classes = np.unique(self.y)
        n_classes = len(classes)
        n_inputs = self.X.shape[1]

        if n_classes == 2:
            n_outputs = 1
            o_act = "sigmoid"
            loss = "bce"
        else:
            n_outputs = n_classes
            o_act = "softmax"
            loss = "cce"

        lr = self.lr_var.get()
        epochs = self.epochs_var.get()
        h_act = self.h_act_var.get()

        model = MLP(
            n_inputs=n_inputs,
            hidden_layers=hidden_layers,
            n_outputs=n_outputs,
            lr=lr,
            epochs=epochs,
            hidden_activation=h_act,
            output_activation=o_act,
            loss=loss,
            seed=42,
        )
        model.fit(self.X, self.y)

        acc = model.score(self.X, self.y)
        final_loss = model.loss_history_[-1]

        info = (
            f"Arquitectura: {[n_inputs] + hidden_layers + [n_outputs]}\n"
            f"Precision : {acc * 100:.2f}%\n"
            f"Loss Final: {final_loss:.5f}\n"
            f"Epocas    : {epochs}"
        )
        self.status_label.config(text=info)

        self.ax_boundary.clear()
        if self.X.shape[1] == 2:
            x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
            y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            grid = np.c_[xx.ravel(), yy.ravel()]
            preds = model.predict(grid).reshape(xx.shape)

            cmap = "coolwarm" if n_classes <= 2 else "tab10"
            self.ax_boundary.contourf(xx, yy, preds, alpha=0.35, cmap=cmap)
            self.ax_boundary.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=cmap, s=50, edgecolors="k", zorder=5)
            self.ax_boundary.set_xlim(x_min, x_max)
            self.ax_boundary.set_ylim(y_min, y_max)

        self.ax_boundary.set_title("Frontera No Lineal del MLP", fontsize=11, fontweight="bold")
        self.ax_boundary.set_xlabel("x1")
        self.ax_boundary.set_ylabel("x2")
        self.ax_boundary.grid(True, linestyle=":", alpha=0.6)

        self.ax_loss.clear()
        self.ax_loss.plot(range(1, len(model.loss_history_) + 1), model.loss_history_, color="#2b5c8f", lw=2)
        self.ax_loss.set_title("Curva de Perdida (Loss)", fontsize=11, fontweight="bold")
        self.ax_loss.set_xlabel("Epoca")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True, linestyle=":", alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw()
