# '''
# Demo of Ant Colony Optimization (ACO) solving a Traveling Salesman Problem (TSP).
# There are many variations of ACO; this is just one approach.
# The problem to solve has a program defined number of cities. We assume that every
# city is connected to every other city. The distance between cities is artificially
# set so that the distance between any two cities is a random value between 1 and 8
# Cities wrap, so if there are 20 cities then D(0,19) = D(19,0).
# Free parameters are alpha, beta, rho, and Q. Hard-coded constants limit min and max
# values of pheromones.
# '''
# from copy  import deepcopy
# import math
# import random
# import sys

# # influence of pheromone on direction
# alpha = 3
# # influence of adjacent node distance
# beta = 2

# # pheromone decrease factor
# rho = 0.01
# # pheromone increase factor
# Q = 2.0

# def main(argv):
#   try:
#     print('\nBegin Ant Colony Optimization demo\n')

#     num_cities = 48
#     num_ants = 4
#     max_time = 100

#     print(f'Number cities in problem = {num_cities}')

#     print(f'\nNumber ants = {num_ants}')
#     print(f'Maximum time = {max_time}')

#     print(f'\nAlpha (pheromone influence) = {alpha}')
#     print(f'Beta (local node influence) = {beta}')
#     print(f'Rho (pheromone evaporation coefficient) = {rho:.2f}')
#     print(f'Q (pheromone deposit factor) = {Q:.2f}')

#     print('\nInitialing dummy graph distances')

#     dists = make_graphDistances(num_cities)

#     print('\nInitialing ants to random trails\n')
#     ants = init_ants(num_ants, num_cities)

#     # initialize ants to random trails
#     show_ants(ants, dists)

#     # determine the best initial trail
#     btrail = best_trail(ants, dists)

#     # the length of the best trail
#     blength = length(btrail, dists)

#     print(f'\nBest initial trail length: {blength:.1f}')
#     #Display(bestTrail)

#     print('\nInitializing pheromones on trails')
#     pheromones = init_pheromones(num_cities)

#     print('\nEntering UpdateAnts - UpdatePheromones loop\n')
#     for time in range(max_time):
#       update_ants(ants, pheromones, dists)
#       update_pheromones(pheromones, ants, dists)

#       curr_bestTrail  = best_trail(ants, dists)
#       curr_bestLength = length(curr_bestTrail, dists)
#       if curr_bestLength < blength:
#         blength = curr_bestLength
#         btrail  = curr_bestTrail
#         print(f'New best length of {blength:.1f} found at time {time}')

#     print('\nTime complete')

#     print('\nBest trail found:')

#     display(btrail)
#     print(f'\nLength of best trail found: {blength:.1f}')

#     print('\nEnd Ant Colony Optimization demo\n')

#   except Exception as ex:
#     print(ex)

# def init_ants(num_ants, num_cities):
#   ants = [0] * num_ants
#   for k in range(num_ants):
#     start = random.randint(0, num_cities)
#     ants[k] = random_trail(start, num_cities)

#   return ants

# def random_trail(start, num_cities):
#   trail = [i for i in range(num_cities)]

#   # Fisher-Yates shuffle
#   for i in range(num_cities):
#     r = random.randint(i, num_cities-1)
#     trail[i], trail[r] = trail[r], trail[i]

#   idx = index_target(trail, start)

#   # put start at [0]
#   trail[0], trail[idx] = trail[idx], trail[0]

#   return trail

# def index_target(trail, target):
#   # helper for RandomTrail
#   for i in range(len(trail)):
#     if trail[i] == target:
#       return i
#   raise Exception('Target not found in IndexOfTarget')

# def length(trail, dists):
#   # total length of a trail
#   result = 0.0
#   for i in range(len(trail)-1):
#     result += distance(trail[i], trail[i + 1], dists)

#   return result

# def best_trail(ants, dists):
#   # best trail has shortest total length
#   blength = length(ants[0], dists)
#   idx_blength = 0
#   for k in range(1, len(ants)):
#     clength = length(ants[k], dists)
#     if clength < blength:
#       blength = clength
#       idx_blength = k
#   num_cities = len(ants[0])

#   return deepcopy(ants[idx_blength])

# def init_pheromones(num_cities):
#   pheromones = [0.0] * num_cities
#   for i in range(num_cities):
#     pheromones[i] = [0.0] * num_cities

#   for i in range(len(pheromones)):
#     for j in range(len(pheromones)):
#       pheromones[i][j] = 0.01;
#       # otherwise first call to UpdateAnts -> BuiuldTrail -> NextNode -> MoveProbs => all 0.0 => throws

#   return pheromones;

# def update_ants(ants, pheromones, dists):
#   num_cities = len(pheromones)
#   for k in range(len(ants)):
#     start = random.randint(0, num_cities-1)
#     ants[k] = build_trail(k, start, pheromones, dists)

# def build_trail(k, start, pheromones, dists):
#   num_cities = len(pheromones)
#   trail = [0] * num_cities
#   visited = [False] * num_cities
#   trail[0] = start
#   visited[start] = True
#   for i in range(num_cities - 1):
#     cityX = trail[i]
#     next_ = next_city(k, cityX, visited, pheromones, dists)
#     trail[i + 1] = next_
#     visited[next_] = True

#   return trail

# def next_city(k, cityX, visited, pheromones, dists):
#   # for ant k (with visited[]), at nodeX, what is next node in trail?
#   probs = move_probs(k, cityX, visited, pheromones, dists)

#   cumul = [0.0] * (len(probs)+1)
#   for i in range(len(probs)):
#     cumul[i + 1] = cumul[i] + probs[i]
#     # consider setting cumul[cuml.Length-1] to 1.00

#   p = random.random()

#   for i in range(len(cumul) - 1):
#     if p >= cumul[i] and p < cumul[i + 1]:
#       return i

#   raise Exception('Failure to return valid city in NextCity')

# def move_probs(k, cityX, visited, pheromones, dists):
#   # for ant k, located at nodeX, with visited[], return the prob of moving to each city
#   num_cities = len(pheromones)
#   taueta = [0.0] * num_cities

#   # inclues cityX and visited cities
#   sum = 0.0
#   # sum of all tauetas
#   # i is the adjacent city
#   for i in range(len(taueta)):
#     if i == cityX:
#       taueta[i] = 0.0
#       # prob of moving to self is 0
#     elif visited[i] == True:
#       taueta[i] = 0.0
#       # prob of moving to a visited city is 0
#     else:
#       taueta[i] = math.pow(pheromones[cityX][i], alpha) * math.pow((1.0 / distance(cityX, i, dists)), beta)
#       # could be huge when pheromone[][] is big
#       if taueta[i] < 0.0001:
#         taueta[i] = 0.0001
#       elif taueta[i] > (sys.float_info.max / (num_cities * 100)):
#         taueta[i] = sys.float_info.max / (num_cities * 100)
#     sum += taueta[i]

#   probs = [0.0] * num_cities
#   for i in range(len(probs)):
#     probs[i] = taueta[i] / sum
#     # big trouble if sum = 0.0

#   return probs

# def update_pheromones(pheromones, ants, dists):
#   for i in range(len(pheromones)):
#     for j in range(i + 1, len(pheromones)):
#       for k in range(len(ants)):
#         clength = length(ants[k], dists)
#         # length of ant k trail
#         decrease = (1.0 - rho) * pheromones[i][j]
#         increase = 0.0
#         if edge_inTrail(i, j, ants[k]) == True:
#           increase = Q / clength

#         pheromones[i][j] = decrease + increase

#         if pheromones[i][j] < 0.0001:
#           pheromones[i][j] = 0.0001
#         elif pheromones[i][j] > 100000.0:
#           pheromones[i][j] = 100000.0

#         pheromones[j][i] = pheromones[i][j]

# def edge_inTrail(cityX, cityY, trail):
#   # are cityX and cityY adjacent to each other in trail[]?
#   last_index = len(trail) - 1
#   idx = index_target(trail, cityX)

#   if idx == 0 and trail[1] == cityY:
#     return True
#   elif idx == 0 and trail[last_index] == cityY:
#     return True
#   elif idx == 0:
#     return False
#   elif idx == last_index and trail[last_index - 1] == cityY:
#     return True
#   elif idx == last_index and trail[0] == cityY:
#     return True
#   elif idx == last_index:
#     return False
#   elif trail[idx - 1] == cityY:
#     return True
#   elif trail[idx + 1] == cityY:
#     return True
#   else:
#     return False

# def make_graphDistances(num_cities):
#   dists = [0.0] * num_cities
#   for i in range(num_cities):
#     dists[i] = [0.0] * num_cities
#   for i in range(num_cities):
#     for j in range(num_cities):
#       d = random.randint(1,9)
#       dists[i][j] = d
#       dists[j][i] = d

#   return dists

# def distance(cityX, cityY, dists):
#   return dists[cityX][cityY]

# def display(trail):
#   for i in range(len(trail)):
#     print('{} '.format(trail[i]), end='')
#     if i > 0 and i % 20 == 0: print()
#   print()

# def show_ants(ants, dists):
#   for i in range(len(ants)):
#     print('{}: ['.format(i), end='')

#     for j in range(4):
#       print('{} '.format(ants[i][j]), end='')
#     print('. . . ', end='')

#     for j in range(len(ants[i])-4, len(ants[i])):
#       print('{} '.format(ants[i][j]), end='')

#     print('] len = {:.1f}'.format(length(ants[i], dists)))

# '''
# def display(pheromones):
#         for i in range(len(pheromones)):
#                 print('{}: '.format(i), end='')
#                 for j in range(len(pheromones)):
#                         print('{:8.4f}'.format(pheromones[i][j], end=''))
#                 print()
# '''

# if __name__ == '__main__':
#   main(sys.argv)



import numpy as np
from .evaluation import tour_length, is_valid_tour


class ACO:
    def __init__(self, m=20, alpha=1.0, beta=5.0, rho=0.3, Q=1.0, tau0=1.0, deposit="best"):
        self.m = int(m)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.Q = float(Q)
        self.tau0 = float(tau0)
        self.deposit = str(deposit)  # "best" o "all"

        self.name = "ACO"
        self.config_id = f"m{self.m}_a{self.alpha}_b{self.beta}_rho{self.rho}_Q{self.Q}_{self.deposit}"

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
            raise ValueError("En este template implementamos 'evals' (recomendado).")
        N = int(budget["N"])

        tau = np.full((n, n), self.tau0, dtype=float)

        eta = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    eta[i, j] = 1.0 / (D[i, j] + 1e-12)

        best = float("inf")
        history = []

        def choose_next(curr, unvisited):
            u = np.array(list(unvisited), dtype=int)
            w = (tau[curr, u] ** self.alpha) * (eta[curr, u] ** self.beta)
            s = float(w.sum())
            if not np.isfinite(s) or s <= 0:
                return int(rng.choice(u))
            p = w / s
            return int(rng.choice(u, p=p))

        def deposit_path(tour, L):
            delta = self.Q / (L + 1e-12)
            for k in range(n):
                i = int(tour[k])
                j = int(tour[(k + 1) % n])
                tau[i, j] += delta
                tau[j, i] += delta

        while evals < N:
            tours = []
            lens = []

            for _ in range(self.m):
                if evals >= N:
                    break

                start = int(rng.integers(0, n))
                unvisited = set(range(n))
                unvisited.remove(start)

                tour = [start]
                curr = start
                while unvisited:
                    nxt = choose_next(curr, unvisited)
                    tour.append(nxt)
                    unvisited.remove(nxt)
                    curr = nxt

                tour = np.array(tour, dtype=int)
                if not is_valid_tour(tour, n):
                    continue

                L = eval_tour(tour)
                tours.append(tour)
                lens.append(L)

            if lens:
                it_best = min(lens)
                if it_best < best:
                    best = it_best
            history.append(float(best))

            # evaporación
            tau *= (1.0 - self.rho)

            # depósito
            if tours:
                if self.deposit == "all":
                    for t, L in zip(tours, lens):
                        deposit_path(t, L)
                else:
                    idx = int(np.argmin(lens))
                    deposit_path(tours[idx], lens[idx])

            tau = np.clip(tau, 1e-12, 1e12)

        return {"best": float(best), "evals_used": int(evals), "history_best": history}