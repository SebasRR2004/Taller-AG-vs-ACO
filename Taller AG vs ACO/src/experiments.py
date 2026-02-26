import time
import pandas as pd
from .evaluation import gap_percent


def run_experiment(algorithm_obj, instance: dict, seed: int, budget: dict) -> dict:
    t0 = time.time()
    out = algorithm_obj.solve(instance, seed=seed, budget=budget)
    t1 = time.time()

    best = out["best"]
    opt = instance.get("opt", None)

    return {
        "best": best,
        "time_sec": t1 - t0,
        "evals_used": out.get("evals_used"),
        "history_best": out.get("history_best"),
        "gap_percent": gap_percent(best, opt),
    }


def multi_seed_runner(algorithms, instances, seeds, budget):
    rows = []
    histories = {}

    for inst in instances:
        for alg in algorithms:
            for sd in seeds:
                res = run_experiment(alg, inst, sd, budget)

                rows.append({
                    "instance": inst["name"],
                    "n": inst["n"],
                    "algorithm": alg.name,
                    "config_id": alg.config_id,
                    "seed": sd,
                    "best": res["best"],
                    "time_sec": res["time_sec"],
                    "evals_used": res["evals_used"],
                    "gap_percent": res["gap_percent"],
                })

                histories[(inst["name"], alg.name, alg.config_id, sd)] = res["history_best"]

    df = pd.DataFrame(rows)
    return df, histories