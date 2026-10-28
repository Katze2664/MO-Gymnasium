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
N_REPEATS = 2
N_TRIALS = 1
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
    xlabel=f"Batch size (batch_size, {N_OBJECTIVES})",
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
        df_impl = df[df["impl"] == name].sort_values("batch_size")
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

        ax.errorbar(df_impl["batch_size"], y, yerr=yerr, label=label, marker="o", capsize=3)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return ax


stmt_baseline_pareto_filter = """
pareto_rewards = []
for _ in range(n_points // batch_size):
    rewards_batch = rng.normal(size=(batch_size, N_OBJECTIVES))
    pareto_rewards.extend(rewards_batch)
rewards_batch = rng.normal(size=(n_points % batch_size, N_OBJECTIVES))
pareto_rewards.extend(rewards_batch)
"""

stmt_baseline_paretoset = """
pareto_batches = []
for _ in range(n_points // batch_size):
    rewards_batch = rng.normal(size=(batch_size, N_OBJECTIVES))
    pareto_batches.append(rewards_batch)
rewards_batch = rng.normal(size=(n_points % batch_size, N_OBJECTIVES))
pareto_batches.append(rewards_batch)
pareto_concat = np.concatenate(pareto_batches, axis=0)
"""

stmt_pareto_filter = """
pareto_rewards = []
for _ in range(n_points // batch_size):
    rewards_batch = rng.normal(size=(batch_size, N_OBJECTIVES))
    pareto_batch = pareto_filter(rewards_batch, minimize=False)
    pareto_rewards.extend(pareto_batch)
rewards_batch = rng.normal(size=(n_points % batch_size, N_OBJECTIVES))
pareto_rewards.extend(rewards_batch)
pareto_rewards = pareto_filter(pareto_rewards, minimize=False)
"""

stmt_paretoset = """
pareto_batches = []
for _ in range(n_points // batch_size):
    rewards_batch = rng.normal(size=(batch_size, N_OBJECTIVES))
    pareto_mask = paretoset(rewards_batch, sense=["max"] * N_OBJECTIVES)
    pareto_batches.append(rewards_batch[pareto_mask])
rewards_batch = rng.normal(size=(n_points % batch_size, N_OBJECTIVES))
pareto_batches.append(rewards_batch)
pareto_concat = np.concatenate(pareto_batches, axis=0)
pareto_mask = paretoset(pareto_concat, sense=["max"] * N_OBJECTIVES)
pareto_rewards = pareto_concat[pareto_mask]
"""

print(f"One-off paretoset load time: {timeit.timeit('paretoset([0.])', globals=globals(), number=1)}")

# name -> (stmt, setup)
impls = {
    "pareto_filter": (stmt_pareto_filter, "pass"),
    "baseline_pareto_filter": (stmt_baseline_pareto_filter, "pass"),
    "paretoset": (stmt_paretoset, "pass"),
    "baseline_paretoset": (stmt_baseline_paretoset, "pass"),
}

n_points = 10**7  # TODO
records = []
for p in range(4, 7):  # TODO
    batch_size = 10**p
    for name, (stmt, setup) in impls.items():
        stats = run(stmt, setup, N_REPEATS, N_TRIALS, seed=42)
        records.append({"n_points": n_points, "batch_size": batch_size, "impl": name, **stats})

df = pd.DataFrame(records)
df.to_csv("pareto_batch_times.csv", index=False)
print(df)

plot(df, stat="mean", errors="std", title="Time to calculate pareto front (x=log, y=log)")
plot(df, stat="mean", errors="std", yscale="linear", title="Time to calculate pareto front (x=log, y=linear)")

print(f"Total time: {time.time() - t0})")
plt.show()
