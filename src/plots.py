import numpy as np
import matplotlib.pyplot as plt


def plot_convergence_mean(histories, instance_name, alg_name, config_id):
    # toma todas las histories de (instance, alg, config) y promedia por posición
    keys = [k for k in histories.keys() if k[0] == instance_name and k[1] == alg_name and k[2] == config_id]
    seqs = [histories[k] for k in keys if histories[k] is not None and len(histories[k]) > 0]
    if not seqs:
        print("No hay histories para esa selección.")
        return

    L = min(len(s) for s in seqs)
    A = np.array([s[:L] for s in seqs], dtype=float)
    mean = A.mean(axis=0)
    std = A.std(axis=0)

    x = np.arange(L)
    plt.figure()
    plt.plot(x, mean)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.title(f"Convergencia promedio - {instance_name} - {alg_name}")
    plt.xlabel("Iteración (registro best-so-far)")
    plt.ylabel("Distancia (best-so-far)")
    plt.show()


def boxplot_best(df, instance_name):
    sub = df[df["instance"] == instance_name]
    algs = sorted(sub["algorithm"].unique())
    data = [sub[sub["algorithm"] == a]["best"].values for a in algs]

    plt.figure()
    plt.boxplot(data, labels=algs)
    plt.title(f"Boxplot best - {instance_name}")
    plt.ylabel("Distancia")
    plt.show()