
# import functools
# import random
# import time
# import math
# import numpy as np
# import matplotlib.pyplot as plt

# from collections import Counter
# from typing import List

# City = complex
# Tour = List[City]

# def distance(A: City, B: City) -> float:
#     return abs(A - B)

# def tour_length(tour: Tour) -> float:
#     return sum(distance(tour[i], tour[i-1]) for i in range(len(tour)))

# def generate_cities(n=10, seed=42):
#     random.seed(seed)
#     return frozenset(
#         City(random.uniform(0,100), random.uniform(0,100))
#         for _ in range(n)
#     )

# cache = functools.lru_cache(None)

# def held_karp(cities):
#     A = next(iter(cities))
#     shortest_segment.cache_clear()

#     return min(
#         (shortest_segment(A, cities - {A, C}, C) for C in cities - {A}),
#         key=tour_length
#     )

# @cache
# def shortest_segment(A, Bs, C):
#     if not Bs:
#         return [A, C]
#     return min(
#         (shortest_segment(A, Bs - {B}, B) + [C] for B in Bs),
#         key=lambda s: sum(abs(s[i]-s[i-1]) for i in range(1,len(s)))
#     )

# def genetic_tsp(cities, pop_size=100, ngen=200):
#     city_list = list(cities)

#     def random_individual():
#         ind = city_list[:]
#         random.shuffle(ind)
#         return ind

#     def crossover(p1, p2):
#         a, b = sorted(random.sample(range(len(p1)), 2))
#         child = [None]*len(p1)
#         child[a:b] = p1[a:b]
#         fill = [c for c in p2 if c not in child]
#         idx = 0
#         for i in range(len(child)):
#             if child[i] is None:
#                 child[i] = fill[idx]
#                 idx += 1
#         return child

#     def mutate(ind):
#         i, j = random.sample(range(len(ind)), 2)
#         ind[i], ind[j] = ind[j], ind[i]

#     population = [random_individual() for _ in range(pop_size)]

#     for _ in range(ngen):
#         population.sort(key=tour_length)
#         next_gen = population[:10]
#         while len(next_gen) < pop_size:
#             p1, p2 = random.sample(population[:50], 2)
#             child = crossover(p1, p2)
#             if random.random() < 0.2:
#                 mutate(child)
#             next_gen.append(child)
#         population = next_gen

#     return min(population, key=tour_length)

# def compare_algorithms(n):
#     cities = generate_cities(n)

#     results = []

#     for algo in [held_karp, genetic_tsp]:
#         start = time.perf_counter()
#         tour = algo(cities)
#         end = time.perf_counter()

#         results.append({
#             "Algorithm": algo.__name__,
#             "Length": tour_length(tour),
#             "Time (s)": round(end-start, 4)
#         })

#     return results

# compare_algorithms(16)


import numpy as np
from .evaluation import tour_length, is_valid_tour


def ox_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator):
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    c1 = -np.ones(n, dtype=int)
    c2 = -np.ones(n, dtype=int)

    c1[a:b] = p1[a:b]
    c2[a:b] = p2[a:b]

    def fill(child, parent):
        pos = b
        for x in parent:
            if x not in child:
                while child[pos % n] != -1:
                    pos += 1
                child[pos % n] = x
                pos += 1
        return child

    return fill(c1, p2), fill(c2, p1)


def inversion_mutation(tour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(tour)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    out = tour.copy()
    out[i:j] = out[i:j][::-1]
    return out


class GA:
    def __init__(self, P=100, pc=0.9, pm=0.05, elitism_k=2, tournament_k=3):
        self.P = int(P)
        self.pc = float(pc)
        self.pm = float(pm)
        self.elitism_k = int(elitism_k)
        self.tournament_k = int(tournament_k)

        self.name = "GA"
        self.config_id = f"P{self.P}_pc{self.pc}_pm{self.pm}_e{self.elitism_k}_t{self.tournament_k}"

    def solve(self, instance: dict, seed: int, budget: dict) -> dict:
        rng = np.random.default_rng(seed)
        D = instance["D"]
        n = instance["n"]

        evals = 0

        def eval_tour(t):
            nonlocal evals
            evals += 1
            return tour_length(t, D)

        # init
        pop = [rng.permutation(n) for _ in range(self.P)]
        fit = [eval_tour(ind) for ind in pop]

        best = min(fit)
        history = [best]

        def tournament_select():
            idx = rng.integers(0, self.P, size=self.tournament_k)
            best_i = min(idx, key=lambda i: fit[i])
            return pop[best_i].copy()

        if budget["type"] == "evals":
            N = int(budget["N"])
        else:
            raise ValueError("En este template implementamos 'evals' (recomendado).")

        while evals < N:
            # elitismo
            elite_idx = np.argsort(fit)[: self.elitism_k]
            new_pop = [pop[i].copy() for i in elite_idx]

            # offspring
            while len(new_pop) < self.P and evals < N:
                p1 = tournament_select()
                p2 = tournament_select()
                c1, c2 = p1, p2

                if rng.random() < self.pc:
                    c1, c2 = ox_crossover(p1, p2, rng)

                if rng.random() < self.pm:
                    c1 = inversion_mutation(c1, rng)
                if rng.random() < self.pm:
                    c2 = inversion_mutation(c2, rng)

                # validación rápida (por si acaso)
                if not is_valid_tour(c1, n) or not is_valid_tour(c2, n):
                    continue

                new_pop.append(c1)
                if len(new_pop) < self.P:
                    new_pop.append(c2)

            pop = new_pop[: self.P]
            fit = [eval_tour(ind) for ind in pop]

            gen_best = min(fit)
            if gen_best < best:
                best = gen_best
            history.append(best)

        return {"best": float(best), "evals_used": int(evals), "history_best": history}