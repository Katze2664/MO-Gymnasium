# %%
import time

t0 = time.time()

import timeit

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from paretoset import paretoset

from mo_gymnasium.envs.minecart.minecart import pareto_filter

N_OBJECTIVES = 3
N_REPEATS = 5
N_TRIALS = 5
print(f"{N_REPEATS=}, {N_TRIALS=}")


def run(stmt, setup, repeat, number, seed=None):
    rng = np.random.default_rng(seed)
    times = (
        np.array(
            timeit.repeat(
                stmt=stmt,
                setup=setup,
                repeat=repeat,
                number=number,
                globals={**globals(), "rng": rng},
            )
        )
        / number
    )  # per-call times
    return {
        "min": times.min(),
        "mean": times.mean(),
        "median": np.median(times),
        "max": times.max(),
        "std": times.std(ddof=1),  # needs repeat >= 2
    }


def plot(
    df,
    impls=None,  # None = all; or a name, or a list of names
    stat="median",  # "min" | "mean" | "median" | "max"
    errors=None,  # None | "range" (min-max) | "std" (± std, ddof=1)
    title="Time to calculate pareto front",
    xlabel=f"Array size (n_points, {N_OBJECTIVES})",
    ylabel="Time (s)",
    xscale="log",
    yscale="log",
    ax=None,
):
    if impls is None:
        impls = df["impl"].unique()
    elif isinstance(impls, str):
        impls = [impls]

    if ax is None:
        _, ax = plt.subplots()

    for name in impls:
        df_impl = df[df["impl"] == name].sort_values("n_points")
        y = df_impl[stat].to_numpy()

        if errors == "range":
            yerr = np.vstack(
                [
                    y - df_impl["min"].to_numpy(),
                    df_impl["max"].to_numpy() - y,
                ]
            )
            label = f"{name} ({stat}, min-max)"
        elif errors == "std":
            yerr = df_impl["std"].to_numpy()
            label = f"{name} ({stat} +/- std)"
        else:
            yerr = None
            label = f"{name} ({stat})"

        ax.errorbar(df_impl["n_points"], y, yerr=yerr, label=label, marker="o", capsize=3)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


setup = """
arr = rng.normal(size=(n_points, N_OBJECTIVES))
"""

stmt_pareto_filter = """
pareto_lst = pareto_filter(arr, minimize=False)
"""

stmt_paretoset = """
mask = paretoset(arr, sense=["max"] * N_OBJECTIVES)
pareto_arr = arr[mask]
"""

print(f"One-off paretoset load time: {timeit.timeit('paretoset([0.])', globals=globals(), number=1)}")

# name -> (stmt, setup)
impls = {
    "pareto_filter": (stmt_pareto_filter, setup),
    "paretoset": (stmt_paretoset, setup),
}

records = []
for p in range(2, 9):
    n_points = 10**p
    print(f"n_points = 10**{p}, Time = {time.time() - t0}")
    for name, (stmt, setup) in impls.items():
        stats = run(stmt, setup, N_REPEATS, N_TRIALS, seed=42)
        records.append({"n_points": n_points, "impl": name, **stats})

df = pd.DataFrame(records)
df.to_csv("pareto_front_times.csv", index=False)
print(df)

plot(df, stat="mean", errors="std", title="Time to calculate pareto front (x=log, y=log)")
plot(df, stat="mean", errors="std", yscale="linear", title="Time to calculate pareto front (x=log, y=linear)")

print(f"Total time: {time.time() - t0}")
plt.show()
