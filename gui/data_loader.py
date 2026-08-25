import csv
import numpy as np


def load_csv_dataset(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","

        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row or not any(row):
                continue
            try:
                numeric_row = [float(val.strip()) for val in row]
                rows.append(numeric_row)
            except ValueError:
                continue

    if len(rows) < 2:
        raise ValueError("El archivo CSV debe contener al menos 2 filas numericas.")

    data = np.array(rows, dtype=float)
    if data.shape[1] < 2:
        raise ValueError("El CSV debe tener al menos una columna de caracteristicas y una columna de etiquetas.")

    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def generate_sample_csv(file_path, dataset_type="circles"):
    from utils.datasets import make_circles, make_moons, make_xor, make_spiral

    if dataset_type == "xor":
        X, y = make_xor()
    elif dataset_type == "moons":
        X, y = make_moons(n_samples=200, noise=0.1)
    elif dataset_type == "spiral":
        X, y = make_spiral(n_samples_per_class=60, n_classes=3, noise=0.1)
    else:
        X, y = make_circles(n_samples=200, noise=0.08, factor=0.5)

    data = np.column_stack([X, y])
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [f"x{i+1}" for i in range(X.shape[1])] + ["label"]
        writer.writerow(header)
        for row in data:
            writer.writerow(row)
