import numpy as np


def is_valid_tour(tour, n: int) -> bool:
    if tour is None:
        return False
    if len(tour) != n:
        return False
    return set(int(x) for x in tour) == set(range(n))


def tour_length(tour, D: np.ndarray) -> float:
    n = len(tour)
    total = 0.0
    for k in range(n):
        i = int(tour[k])
        j = int(tour[(k + 1) % n])
        total += float(D[i, j])
    return total


def gap_percent(L_alg: float, L_opt: float | None) -> float | None:
    """
    GAP (%) = (L_alg - L_opt) / L_opt * 100
    Si no hay óptimo conocido, retorna None.
    """
    if L_opt is None:
        return None
    if L_opt == 0:
        return None
    return (L_alg - L_opt) / L_opt * 100.0