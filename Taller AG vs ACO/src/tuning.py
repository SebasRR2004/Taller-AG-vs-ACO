import itertools


def make_ga_grid():
    # Grid pequeño (informativo, no enorme)
    P_vals = [50, 100, 200]
    pc_vals = [0.85, 0.95]
    pm_vals = [0.03, 0.05, 0.10]
    elit_vals = [1, 2, 5]
    tour_vals = [3]  # fijo para no explotar combinaciones

    grid = []
    for P, pc, pm, e, t in itertools.product(P_vals, pc_vals, pm_vals, elit_vals, tour_vals):
        grid.append({"P": P, "pc": pc, "pm": pm, "elitism_k": e, "tournament_k": t})
    return grid


def make_aco_grid():
    m_vals = [10, 20, 40]
    a_vals = [0.5, 1.0, 2.0]
    b_vals = [2.0, 5.0, 8.0]
    rho_vals = [0.1, 0.3, 0.5]
    deposit_vals = ["best"]  # fijo para estabilidad; luego pruebas "all"
    Q_vals = [1.0]

    grid = []
    for m, a, b, rho, dep, Q in itertools.product(m_vals, a_vals, b_vals, rho_vals, deposit_vals, Q_vals):
        grid.append({"m": m, "alpha": a, "beta": b, "rho": rho, "deposit": dep, "Q": Q})
    return grid


def make_cbga_grid():
    P_vals = [50, 100]
    pc_vals = [0.85, 0.95]
    pm_vals = [0.03, 0.05, 0.10]
    div_vals = [0.15, 0.25, 0.35]  # bajo/medio/alto
    rep_vals = ["worst", "most_similar", "worst_similar"]
    ls_vals = ["off", "child"]  # luego puedes probar top5/top10

    grid = []
    for P, pc, pm, div, rep, ls in itertools.product(P_vals, pc_vals, pm_vals, div_vals, rep_vals, ls_vals):
        grid.append({
            "P": P, "pc": pc, "pm": pm,
            "diversity_threshold": div,
            "replacement": rep,
            "use_2opt": ls
        })
    return grid