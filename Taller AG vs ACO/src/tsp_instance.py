import os
import numpy as np
import tsplib95


def load_tsp(path: str, opt_known: float | None = None) -> dict:
    """
    Carga una instancia TSPLIB .tsp y crea matriz de distancias D (n x n).
    opt_known: óptimo conocido si lo tienes (si no, None).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe: {path}")

    problem = tsplib95.load(path)
    n = int(problem.dimension)

    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = float(problem.get_weight(i + 1, j + 1))

    name = getattr(problem, "name", os.path.basename(path))
    return {"name": name, "path": path, "n": n, "D": D, "opt": opt_known}