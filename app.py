import tkinter as tk
from tkinter import ttk

from gui.tab_neuron import NeuronSimulatorTab
from gui.tab_perceptron import PerceptronTrainerTab
from gui.tab_mlp import MLPTrainerTab


class NeuroLearnNetApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NeuroLearnNet Studio - Simulador Visual de Redes Neuronales")
        self.geometry("1100, 680")
        self.minsize(900, 580)
        self._setup_theme()
        self._build_tabs()

    def _setup_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", font=("Segoe UI", 9))
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[12, 6])
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=4)
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground="#2b5c8f")

    def _build_tabs(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        tab_neuron = NeuronSimulatorTab(notebook)
        notebook.add(tab_neuron, text=" ⚡ 1. Simulador de Neurona ")

        tab_perceptron = PerceptronTrainerTab(notebook)
        notebook.add(tab_perceptron, text=" 📐 2. Perceptron Monocapa ")

        tab_mlp = MLPTrainerTab(notebook)
        notebook.add(tab_mlp, text=" 🧠 3. Red Neuronal MLP / Datos CSV ")


def main():
    app = NeuroLearnNetApp()
    app.mainloop()


if __name__ == "__main__":
    main()
