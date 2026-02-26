import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["instance", "n", "algorithm", "config_id"])
              .agg(best_mean=("best", "mean"),
                   best_std=("best", "std"),
                   best_min=("best", "min"),
                   best_max=("best", "max"),
                   time_mean=("time_sec", "mean"),
                   evals_mean=("evals_used", "mean"))
              .reset_index())


def rank_configs(summary_df: pd.DataFrame, w_quality=1.0, w_stability=0.2, w_time=0.05) -> pd.DataFrame:
    """
    Score menor = mejor.
    Normalizamos por instancia+algoritmo para que no se mezcle escalas raras.
    """
    rows = []
    for (inst, alg), sub in summary_df.groupby(["instance", "algorithm"]):
        s = sub.copy()

        # normalizaciones (evitar dividir por 0)
        qmin, qmax = s["best_mean"].min(), s["best_mean"].max()
        stmin, stmax = s["best_std"].min(), s["best_std"].max()
        tmin, tmax = s["time_mean"].min(), s["time_mean"].max()

        def norm(x, lo, hi):
            if hi - lo < 1e-12:
                return 0.0
            return (x - lo) / (hi - lo)

        s["q_norm"] = s["best_mean"].apply(lambda x: norm(x, qmin, qmax))
        s["st_norm"] = s["best_std"].apply(lambda x: norm(x, stmin, stmax))
        s["t_norm"] = s["time_mean"].apply(lambda x: norm(x, tmin, tmax))

        s["score"] = w_quality * s["q_norm"] + w_stability * s["st_norm"] + w_time * s["t_norm"]
        rows.append(s)

    ranked = pd.concat(rows, ignore_index=True)
    ranked = ranked.sort_values(["instance", "algorithm", "score"], ascending=[True, True, True])
    return ranked


def best_config_per_instance(ranked_df: pd.DataFrame) -> pd.DataFrame:
    return (ranked_df.groupby(["instance", "algorithm"])
                     .head(1)
                     .reset_index(drop=True))

