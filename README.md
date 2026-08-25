# NeuroLearnNet 🧠⚡

**NeuroLearnNet** es una biblioteca educativa, práctica y visual para comprender, simular y construir **Redes Neuronales Artificiales desde cero** usando únicamente **Python y NumPy**.

El proyecto abarca desde los fundamentos biológicos de la neurona y el Perceptrón monocapa hasta redes neuronales profundas (*Multilayer Perceptron - MLP*) entrenadas con *Backpropagation* (retropropagación del error), complementado con una **interfaz gráfica interactiva de escritorio**.

---

## 📑 Tabla de Contenidos
1. [Inspiración Biológica](#1-inspiración-biológica)
2. [El Perceptrón Monocapa](#2-el-perceptrón-monocapa)
3. [El Problema de la No-Linealidad (XOR)](#3-el-problema-de-la-no-linealidad-xor)
4. [Redes Neuronales Multicapa (MLP)](#4-redes-neuronales-multicapa-mlp)
5. [Algoritmo de Backpropagation](#5-algoritmo-de-backpropagation)
6. [Funciones de Activación y Pérdida](#6-funciones-de-activación-y-pérdida)
7. [Interfaz Gráfica Interactiva (NeuroLearnNet Studio)](#7-interfaz-gráfica-interactiva-neurolearnnet-studio)
8. [Estructura del Repositorio](#8-estructura-del-repositorio)
9. [Instalación y Requisitos](#9-instalación-y-requisitos)
10. [Simulaciones y Experimentos](#10-simulaciones-y-experimentos)
11. [Carga de Datasets Propios (CSV o Clics)](#11-carga-de-datasets-propios-csv-o-clics)
12. [Ejecución de Pruebas Unitarias](#12-ejecución-de-pruebas-unitarias)

---

## 1. Inspiración Biológica

Las redes neuronales artificiales nacen como un modelo matemático simplificado del sistema nervioso humano.

```
       Dendritas                Soma               Axón          Terminales Sinápticas
 (Entradas / Estímulos)  ->  (Integración)  ->  (Conducción)  ->    (Transmisión)
        [ x1, x2 ]               Σ w·x + b            a(z)             Salida (y)
```

* **Dendritas (Entradas $x_i$):** Reciben señales electroquímicas de neuronas previas.
* **Sinapsis (Pesos sinápticos $w_i$):** Modulan la fuerza y relevancia de cada conexión (excitatoria o inhibitoria).
* **Soma (Suma ponderada $z$):** Integra las señales acumuladas agregando el sesgo (*bias* $b$).
* **Potencial de Acción (Función de Activación $\sigma(z)$):** Si la suma supera cierto umbral, la neurona se dispara generando una respuesta enviada a través del **axón**.

![Neurona Biologica vs Artificial](https://github.com/user-attachments/assets/d08d9449-ad16-41ba-b3c7-bbf7f9dd2768)

---

## 2. El Perceptrón Monocapa

Propuesto por Frank Rosenblatt en 1957, el **Perceptrón** es la unidad computacional fundamental de clasificación binaria lineal.

![Perceptron](https://github.com/user-attachments/assets/96689f40-2d0a-495f-9613-92e4df353dca)

### Formulación Matemática

Dado un vector de características $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$:

1. **Combinación Lineal (Pre-activación):**
   $$z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^n w_i x_i + b$$

2. **Función Escalón (Heaviside Step Function):**
   $$\hat{y} = \begin{cases} 1 & \text{si } z \ge 0 \\ 0 & \text{si } z < 0 \end{cases}$$

3. **Regla de Actualización de Pesos (Regla del Perceptrón):**
   $$w_i \leftarrow w_i + \eta \cdot (y - \hat{y}) \cdot x_i$$
   $$b \leftarrow b + \eta \cdot (y - \hat{y})$$
   donde $\eta$ es la tasa de aprendizaje (*learning rate*).

![Arquitectura Perceptron](https://github.com/user-attachments/assets/0f31292f-14c4-438a-90ed-269c8b5fded7)

---

## 3. El Problema de la No-Linealidad (XOR)

En 1969, Marvin Minsky y Seymour Papert demostraron que un perceptrón monocapa **solo puede resolver problemas linealmente separables** (como compuertas AND, OR y NAND).

La compuerta **XOR (OR Exclusivo)** no es linealmente separable:

| $x_1$ | $x_2$ | AND | OR | NAND | **XOR** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 0 | 1 | **0** |
| 0 | 1 | 0 | 1 | 1 | **1** |
| 1 | 0 | 0 | 1 | 1 | **1** |
| 1 | 1 | 1 | 1 | 0 | **0** |

Para resolver XOR y fronteras no lineales complejas, se requiere introducir **capas ocultas** con funciones de activación no lineales.

---

## 4. Redes Neuronales Multicapa (MLP)

Un **Perceptrón Multicapa (MLP)** consiste en una capa de entrada, una o más capas ocultas y una capa de salida.

```
 Capa Entrada (x)         Capa Oculta (a1)         Capa de Salida (a2)
     (x1)  ---------\ /-------- (h1) --------\ /-------- (y_hat)
                     X                        X
     (x2)  ---------/ \-------- (h2) --------/ \--------
```

### Paso Hacia Adelante (Forward Pass)

Para cada capa $l = 1, \dots, L$:
$$Z^{(l)} = A^{(l-1)} W^{(l)} + b^{(l)}$$
$$A^{(l)} = \sigma^{(l)}(Z^{(l)})$$
donde $A^{(0)} = X$.

---

## 5. Algoritmo de Backpropagation

El algoritmo de retropropagación del error (*Backpropagation*) utiliza la **Regla de la Cadena** para calcular el gradiente de la función de pérdida con respecto a todos los pesos y sesgos de la red:

1. **Gradiente en la Capa de Salida ($L$):**
   $$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial Z^{(L)}} = \frac{\partial \mathcal{L}}{\partial A^{(L)}} \odot {\sigma^{(L)}}'(Z^{(L)})$$

2. **Retropropagación hacia Capas Ocultas ($l$):**
   $$\delta^{(l)} = \left( \delta^{(l+1)} (W^{(l+1)})^T \right) \odot {\sigma^{(l)}}'(Z^{(l)})$$

3. **Cálculo de Gradientes:**
   $$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{1}{m} (A^{(l-1)})^T \delta^{(l)}$$
   $$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \frac{1}{m} \sum_{i=1}^m \delta^{(l)}$$

4. **Actualización por Descenso de Gradiente:**
   $$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial W^{(l)}}$$
   $$b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial \mathcal{L}}{\partial b^{(l)}}$$

---

## 6. Funciones de Activación y Pérdida

### Activaciones Implementadas
* **Sigmoide:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
* **ReLU:** $f(z) = \max(0, z)$
* **LeakyReLU:** $f(z) = \max(\alpha z, z)$
* **Tanh:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
* **GELU:** $f(z) = \frac{z}{2}\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(z + 0.044715 z^3\right)\right)\right]$
* **Softmax:** $P(y=k \mid \mathbf{z}) = \frac{e^{z_k}}{\sum_j e^{z_j}}$
* **Linear:** $f(z) = z$
* **Step:** $f(z) = \mathbb{I}(z \ge 0)$

### Funciones de Pérdida
* **MSE (Error Cuadrático Medio):** $\mathcal{L} = \frac{1}{m}\sum (y - \hat{y})^2$
* **MAE (Error Absoluto Medio):** $\mathcal{L} = \frac{1}{m}\sum |y - \hat{y}|$
* **Binary Cross Entropy (BCE):** $\mathcal{L} = -\frac{1}{m}\sum [y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$
* **Categorical Cross Entropy (CCE):** $\mathcal{L} = -\frac{1}{m}\sum \sum y_{ik} \ln \hat{y}_{ik}$

---

## 7. Interfaz Gráfica Interactiva (NeuroLearnNet Studio)

El proyecto cuenta con una aplicación gráfica de escritorio que permite experimentar de forma visual sin escribir código:

```bash
python app.py
```

### Funcionalidades de la Interfaz:
1. **⚡ Simulador de Neurona:**
   * Modifica en tiempo real entradas ($x_1, x_2$), pesos ($w_1, w_2$) y sesgo ($b$) mediante controles deslizantes (*sliders*).
   * Visualiza el diagrama neuronal interactivo y la posición de disparo en la curva de activación.
2. **📐 Entrenador de Perceptrón:**
   * Entrena compuertas lógicas (AND, OR, NAND) o dibuja tus propios puntos en el plano 2D haciendo clic.
   * Visualiza la recta separadora y la disminución de errores por época.
3. **🧠 Entrenador de MLP y Datasets Personalizados:**
   * Configura libremente la arquitectura de capas ocultas (ejemplo: `32, 16, 8`), funciones de activación, tasa de aprendizaje y épocas.
   * Entrena sobre datasets clásicos (XOR, Círculos, Lunas, Espirales) o importa tus propios archivos `.csv`.
   * Visualiza el mapa de calor de la frontera de decisión no lineal y la curva de pérdida (*loss*).

---

## 8. Estructura del Repositorio

```text
NeuroLearnNet/
├── app.py                    # Aplicacion grafica de escritorio
├── gui/
│   ├── __init__.py           # Exportacion de componentes graficos
│   ├── data_loader.py        # Importador y exportador de archivos CSV
│   ├── tab_neuron.py         # Pestana 1: Simulador interactivo de neurona
│   ├── tab_perceptron.py     # Pestana 2: Entrenador visual de Perceptron
│   └── tab_mlp.py            # Pestana 3: Entrenador visual de MLP
├── models/
│   ├── __init__.py           # Exportacion de modelos
│   ├── perceptron.py         # Implementacion del Perceptron Simple
│   └── mlp.py                # Implementacion del MLP Multicapa
├── utils/
│   ├── __init__.py           # Exportacion de utilidades
│   ├── activation.py         # Funciones de activacion y sus derivadas
│   ├── loss.py               # Funciones de perdida y sus derivadas
│   ├── datasets.py           # Generadores de datasets sinteticos
│   └── visualization.py      # Graficadores de fronteras y curvas
├── experiments/
│   ├── simulate_neuron.py    # Simulacion en consola de neurona
│   ├── train_perceptron.py   # Entrenamiento en compuertas logicas
│   ├── train_xor.py          # Comparativa Perceptron vs MLP en XOR
│   └── train_mlp.py          # Entrenamiento no lineal y multiclase
├── tests/
│   ├── test_activations.py   # Pruebas unitarias de activaciones
│   ├── test_losses.py        # Pruebas unitarias de perdidas
│   ├── test_perceptron.py    # Pruebas unitarias de Perceptron
│   └── test_mlp.py           # Pruebas unitarias de MLP
├── pytest.ini                # Configuracion de test runner
├── requirements.txt          # Dependencias del proyecto
└── README.md                 # Documentacion principal
```

---

## 9. Instalación y Requisitos

Requiere Python 3.8 o superior.

```bash
git clone https://github.com/Mik0-T3ch/NeuroLearnNet.git
cd NeuroLearnNet
pip install -r requirements.txt
```

---

## 10. Simulaciones y Experimentos

También puedes ejecutar los experimentos por línea de comandos:

### 1. Simular una Neurona Individual
```bash
python experiments/simulate_neuron.py
```

### 2. Entrenar Perceptrón en Compuertas Lógicas
```bash
python experiments/train_perceptron.py
```

### 3. Resolver el Problema XOR (Perceptrón vs MLP)
```bash
python experiments/train_xor.py
```

### 4. Entrenar MLP en Patrones No Lineales y Multiclase
```bash
python experiments/train_mlp.py
```

---

## 11. Carga de Datasets Propios (CSV o Clics)

La aplicación soporta entrenar redes neuronales con tus propios datos:

### Formato de Archivo CSV
Crea un archivo `.csv` donde las primeras columnas sean las características numéricas y la última columna sea la etiqueta de clase (`0`, `1`, `2`, ...):

```csv
x1,x2,label
1.25,-0.45,0
-0.80,1.10,1
0.50,0.90,1
```

En la interfaz gráfica, selecciona **Cargar Archivo CSV** y haz clic en **Seleccionar CSV...**. También puedes presionar **Generar Ejemplo CSV** para obtener una plantilla lista para usar.

---

## 12. Ejecución de Pruebas Unitarias

Para verificar la consistencia matemática de todo el proyecto:

```bash
pytest
```
