import os

from src.tsp_instance import load_tsp
from src.ga import GA
from src.aco import ACO
from src.cbga import CBGA
from src.experiments import multi_seed_runner
from src.analysis_tables import summarize, rank_configs, best_config_per_instance
from src.plots import plot_convergence_mean, boxplot_best
from src.tuning import make_ga_grid, make_aco_grid, make_cbga_grid

def main():
    # import os
    # print("Working directory:", os.getcwd())
    # 0) carpetas
    os.makedirs("results", exist_ok=True)
    print("Working directory:", os.getcwd())
    # 1) Cargar instancias
    print(">> Voy a cargar instancias...", flush=True)

    instances = [
        (print(">> Cargando eil51...", flush=True), load_tsp("data/eil51.tsp", opt_known=426))[1],
        (print(">> Cargando berlin52...", flush=True), load_tsp("data/berlin52.tsp", opt_known=7542))[1],
        (print(">> Cargando st70...", flush=True), load_tsp("data/st70.tsp", opt_known=675))[1],
    ]

    print(">> Instancias cargadas OK", flush=True)

    # # 2) Presupuesto justo (evals)
    # budget = {"type": "evals", "N": 50000}

    # # Seeds
    # seeds_baseline = list(range(30))
    # seeds_tuning = list(range(15))
    # seeds_final = list(range(30))

    # 2) Presupuesto justo (evals)  -> modo PRUEBA
    budget = {"type": "evals", "N": 1}

    # Seeds -> modo PRUEBA
    seeds_baseline = list(range(1))
    seeds_tuning = list(range(1))
    seeds_final = list(range(1))

    # 3) BASELINE
    ga = GA(P=100, pc=0.9, pm=0.05, elitism_k=2, tournament_k=3)
    aco = ACO(m=20, alpha=1.0, beta=5.0, rho=0.3, Q=1.0, deposit="best")
    cbga = CBGA(
        P=100,
        pc=0.9,
        pm=0.05,
        diversity_threshold=0.25,
        replacement="worst_similar",
        use_2opt="off"
    )

    algorithms_baseline = [ga, aco, cbga]

    df_baseline, histories_baseline = multi_seed_runner(
        algorithms_baseline, instances, seeds_baseline, budget
    )
    df_baseline.to_csv("results/baseline_runs.csv", index=False)

    summary_baseline = summarize(df_baseline).sort_values(["instance", "algorithm"])
    summary_baseline.to_csv("results/baseline_summary.csv", index=False)

    # plots baseline
    for inst in instances:
        boxplot_best(df_baseline, inst["name"])
    for inst in instances:
        for alg in algorithms_baseline:
            plot_convergence_mean(histories_baseline, inst["name"], alg.name, alg.config_id)

    # 4) TUNING (grid)
    ga_cfgs = make_ga_grid()
    aco_cfgs = make_aco_grid()
    cbga_cfgs = make_cbga_grid()

    algorithms_tuning = []
    for cfg in ga_cfgs:
        algorithms_tuning.append(GA(**cfg))
    for cfg in aco_cfgs:
        algorithms_tuning.append(ACO(**cfg))
    for cfg in cbga_cfgs:
        algorithms_tuning.append(CBGA(**cfg))

    # tuning con 2 instancias para acelerar
    instances_tuning = instances[:2]

    df_tune, histories_tune = multi_seed_runner(
        algorithms_tuning, instances_tuning, seeds_tuning, budget
    )
    df_tune.to_csv("results/tuning_runs.csv", index=False)

    summary_tune = summarize(df_tune)
    ranked = rank_configs(summary_tune, w_quality=1.0, w_stability=0.2, w_time=0.05)
    best_per = best_config_per_instance(ranked)
    best_per.to_csv("results/tuning_best_per_instance.csv", index=False)

    # 5) Armar lista de mejores algoritmos
    best_algorithms = []
    for _, row in best_per.iterrows():
        alg = row["algorithm"]
        config_id = row["config_id"]
        obj = next(a for a in algorithms_tuning if a.name == alg and a.config_id == config_id)
        best_algorithms.append(obj)

    # 6) VALIDACIÓN FINAL
    df_final, histories_final = multi_seed_runner(
        best_algorithms, instances, seeds_final, budget
    )
    df_final.to_csv("results/final_runs.csv", index=False)

    # summary_final = summarize(df_final).sort_values(["instance", "algorithm"])
    # summary_final.to_csv("results/final_summary.csv", index=False)

    summary_final = summarize(df_final)

    # Calcular GAP promedio por instancia y algoritmo
    gap_mean = (
        df_final.groupby(["instance", "algorithm"])["gap_percent"]
        .mean()
        .reset_index(name="gap_mean")
    )

    # Unir tablas
    summary_final = summary_final.merge(
        gap_mean,
        on=["instance", "algorithm"],
        how="left"
    )

    # Ordenar por mejor calidad
    summary_final = summary_final.sort_values(["instance", "best_mean"])

    # Guardar versión completa tipo paper
    summary_final.to_csv("results/final_summary_with_gap.csv", index=False)

    print("\n=== TABLA FINAL TIPO PAPER ===")
    print(summary_final)

    # plots finales
    for inst in instances:
        boxplot_best(df_final, inst["name"])
    for inst in instances:
        for alg in best_algorithms:
            plot_convergence_mean(histories_final, inst["name"], alg.name, alg.config_id)


if __name__ == "__main__":
    main()