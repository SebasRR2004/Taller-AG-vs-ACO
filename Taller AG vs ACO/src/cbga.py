import numpy as np
from .evaluation import tour_length, is_valid_tour
from .ga import ox_crossover, inversion_mutation
from .local_search import two_opt


def edges_of(tour: np.ndarray):
    n = len(tour)
    E = set()
    for k in range(n):
        a = int(tour[k])
        b = int(tour[(k + 1) % n])
        if a < b:
            E.add((a, b))
        else:
            E.add((b, a))
    return E


def edge_distance(t1: np.ndarray, t2: np.ndarray) -> float:
    E1 = edges_of(t1)
    E2 = edges_of(t2)
    inter = len(E1 & E2)
    return 1.0 - inter / len(E1)


def tour_hash(tour: np.ndarray) -> tuple:
    # hash estable: tupla de ints
    return tuple(int(x) for x in tour)


class CBGA:
    def __init__(
        self,
        P=100,
        pc=0.9,
        pm=0.05,
        diversity_threshold=0.25,  # 0.0..1.0 (más alto = más estricto)
        replacement="worst_similar",  # "worst", "most_similar", "worst_similar"
        use_2opt="off",  # "off", "top5", "top10", "child"
        tournament_k=3,
        elitism_k=2,
    ):
        self.P = int(P)
        self.pc = float(pc)
        self.pm = float(pm)
        self.diversity_threshold = float(diversity_threshold)
        self.replacement = str(replacement)
        self.use_2opt = str(use_2opt)
        self.tournament_k = int(tournament_k)
        self.elitism_k = int(elitism_k)

        self.name = "CBGA"
        self.config_id = (
            f"P{self.P}_pc{self.pc}_pm{self.pm}_div{self.diversity_threshold}_"
            f"rep{self.replacement}_2opt{self.use_2opt}"
        )

    def solve(self, instance: dict, seed: int, budget: dict) -> dict:
        rng = np.random.default_rng(seed)
        D = instance["D"]
        n = instance["n"]

        evals = 0
        def eval_tour(t):
            nonlocal evals
            evals += 1
            return tour_length(t, D)

        if budget["type"] != "evals":
            raise ValueError("En este template implementamos 'evals'.")
        N = int(budget["N"])

        # init población sin duplicados
        pop = []
        pop_set = set()
        while len(pop) < self.P:
            t = rng.permutation(n)
            h = tour_hash(t)
            if h not in pop_set:
                pop.append(t)
                pop_set.add(h)

        fit = [eval_tour(ind) for ind in pop]
        best = min(fit)
        history = [float(best)]

        def tournament_select():
            idx = rng.integers(0, self.P, size=self.tournament_k)
            best_i = min(idx, key=lambda i: fit[i])
            return pop[best_i].copy()

        def diversity_ok(child):
            # child debe ser suficientemente diferente de TODOS (o del más parecido)
            dmin = min(edge_distance(child, p) for p in pop)
            return dmin >= self.diversity_threshold, dmin

        def pick_replacement_index(child):
            if self.replacement == "worst":
                return int(np.argmax(fit))

            # más similar
            dists = [edge_distance(child, p) for p in pop]
            most_similar = int(np.argmin(dists))

            if self.replacement == "most_similar":
                return most_similar

            # worst_similar: entre los k más similares, saca el peor
            k = max(5, self.P // 10)
            sim_idx = np.argsort(dists)[:k]
            worst_among = int(sim_idx[np.argmax([fit[i] for i in sim_idx])])
            return worst_among

        # loop principal por evals
        while evals < N:
            # generar hijo
            p1 = tournament_select()
            p2 = tournament_select()
            child = p1

            if rng.random() < self.pc:
                c1, _ = ox_crossover(p1, p2, rng)
                child = c1

            if rng.random() < self.pm:
                child = inversion_mutation(child, rng)

            if not is_valid_tour(child, n):
                continue

            # opcional: 2-opt al hijo
            if self.use_2opt == "child":
                child = two_opt(child, D, max_passes=30)

            h = tour_hash(child)
            if h in pop_set:
                # no duplicados
                continue

            # diversidad
            ok, _ = diversity_ok(child)
            if not ok:
                continue

            child_fit = eval_tour(child)
            # debe mejorar a alguien
            rep_i = pick_replacement_index(child)
            if child_fit + 1e-12 < fit[rep_i]:
                # reemplazar
                pop_set.remove(tour_hash(pop[rep_i]))
                pop[rep_i] = child
                fit[rep_i] = child_fit
                pop_set.add(h)

            # intensificación: 2-opt top-k (opcional)
            if self.use_2opt in ("top5", "top10") and evals < N:
                topk = 5 if self.use_2opt == "top5" else 10
                idxs = np.argsort(fit)[:topk]
                for i in idxs:
                    improved = two_opt(pop[i], D, max_passes=20)
                    ih = tour_hash(improved)
                    if ih != tour_hash(pop[i]) and ih not in pop_set:
                        pop_set.remove(tour_hash(pop[i]))
                        pop[i] = improved
                        fit[i] = eval_tour(improved)
                        pop_set.add(ih)
                    if evals >= N:
                        break

            cur_best = min(fit)
            if cur_best < best:
                best = cur_best
            history.append(float(best))

        return {"best": float(best), "evals_used": int(evals), "history_best": history}