import numpy as np


def make_and():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1], dtype=float)
    return X, y


def make_or():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 1], dtype=float)
    return X, y


def make_nand():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([1, 1, 1, 0], dtype=float)
    return X, y


def make_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)
    return X, y


def make_linearly_separable(n_samples=100, seed=None):
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2
    
    class_0 = rng.normal(loc=[-1.5, -1.5], scale=0.6, size=(n_per_class, 2))
    class_1 = rng.normal(loc=[1.5, 1.5], scale=0.6, size=(n_per_class, 2))
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    indices = rng.permutation(len(X))
    return X[indices], y[indices]


def make_circles(n_samples=200, noise=0.05, factor=0.5, seed=None):
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2
    
    angles_outer = rng.uniform(0, 2 * np.pi, n_per_class)
    r_outer = 1.0 + rng.normal(0, noise, n_per_class)
    X_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    
    angles_inner = rng.uniform(0, 2 * np.pi, n_per_class)
    r_inner = factor + rng.normal(0, noise, n_per_class)
    X_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    X = np.vstack([X_outer, X_inner])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    indices = rng.permutation(len(X))
    return X[indices], y[indices]


def make_moons(n_samples=200, noise=0.1, seed=None):
    rng = np.random.default_rng(seed)
    n_per_class = n_samples // 2
    
    theta_0 = rng.uniform(0, np.pi, n_per_class)
    x_0 = np.cos(theta_0) + rng.normal(0, noise, n_per_class)
    y_0 = np.sin(theta_0) + rng.normal(0, noise, n_per_class)
    class_0 = np.column_stack([x_0, y_0])
    
    theta_1 = rng.uniform(0, np.pi, n_per_class)
    x_1 = 1.0 - np.cos(theta_1) + rng.normal(0, noise, n_per_class)
    y_1 = 0.5 - np.sin(theta_1) + rng.normal(0, noise, n_per_class)
    class_1 = np.column_stack([x_1, y_1])
    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    indices = rng.permutation(len(X))
    return X[indices], y[indices]


def make_spiral(n_samples_per_class=100, n_classes=3, noise=0.2, seed=None):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples_per_class * n_classes, 2))
    y = np.zeros(n_samples_per_class * n_classes, dtype=int)
    
    for j in range(n_classes):
        ix = range(n_samples_per_class * j, n_samples_per_class * (j + 1))
        r = np.linspace(0.0, 1.0, n_samples_per_class)
        t = np.linspace(j * 4.0, (j + 1) * 4.0, n_samples_per_class) + rng.normal(0, noise, n_samples_per_class)
        X[ix] = np.column_stack([r * np.sin(t), r * np.cos(t)])
        y[ix] = j
        
    indices = rng.permutation(len(X))
    return X[indices], y[indices]
