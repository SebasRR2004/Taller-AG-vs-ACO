import numpy as np
from .evaluation import tour_length


def two_opt(tour: np.ndarray, D: np.ndarray, max_passes: int = 50) -> np.ndarray:
    """
    2-opt clásico: mejora por inversión de segmentos.
    max_passes limita ciclos.
    """
    best = tour.copy()
    best_len = tour_length(best, D)
    n = len(best)

    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                if j - i == 1:
                    continue
                candidate = best.copy()
                candidate[i:j] = candidate[i:j][::-1]
                cand_len = tour_length(candidate, D)
                if cand_len + 1e-12 < best_len:
                    best = candidate
                    best_len = cand_len
                    improved = True
        # si quieres acelerar: break si improved y reiniciar loops; aquí lo dejamos simple
    return best