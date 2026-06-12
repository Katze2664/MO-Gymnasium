
# Compares time taken to run different pareto front versions
# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd

import mo_gymnasium.envs.minecart.minecart1 as mc_1
import mo_gymnasium.envs.minecart.minecart2 as mc_2
import mo_gymnasium.envs.minecart.minecart3 as mc_3
import mo_gymnasium.envs.minecart.minecart4 as mc_4
import mo_gymnasium.envs.minecart.minecart5 as mc_5
import mo_gymnasium.envs.minecart.minecart6 as mc_6
import mo_gymnasium.envs.minecart.minecart7 as mc_7


config_stochastic = Path(r"C:\Users\ethan\Desktop\Cloned repos\MO-Gymnasium\mo_gymnasium\envs\minecart\mine_config.json")
config_det = Path(r"C:\Users\ethan\Desktop\Cloned repos\MO-Gymnasium\mo_gymnasium\envs\minecart\mine_config_det.json")

frame_skip = 4
gamma = 0.98
config = config_det
incremental = True
symmetric = False

mcs = [
    ("minecart1", mc_1,),
    ("minecart2", mc_2,),
    ("minecart3", mc_3,),
    ("minecart4", mc_4,),
    ("minecart5", mc_5,),
    ("minecart6", mc_6,),
    ("minecart7", mc_7,),
]
minecarts = []
for name, mc in mcs:
    minecart = mc.Minecart(
        frame_skip=frame_skip,
        incremental_frane_skip=incremental,
        config=config,
    )
    minecarts.append((name, minecart))

replicates = 5
batch_sizes = [10**3, 10**4, 10**5, 10**6, 10**7]
records = []  # each entry: {"Version": ..., "Batch Size": ..., "Rep": rep, "Time": t}

for rep in range(replicates):
    name, minecart = minecarts[0]
    time_zero = time.time()
    pareto_front_1 = minecart.pareto_front(
        gamma=gamma,
        symmetric=symmetric,
    )
    t = time.time() - time_zero
    print("{name} No_batch", t)
    records.append({"Version": name, "Batch Size": min(batch_sizes), "Rep": rep, "Time": t})
    records.append({"Version": name, "Batch Size": max(batch_sizes), "Rep": rep, "Time": t})
    assert isinstance(pareto_front_1, list)
    pareto_rewards_1 = np.array(pareto_front_1)

    for batch_size in batch_sizes:
        for name, minecart in minecarts[1:]:
            time_zero = time.time()
            pareto_front = minecart.pareto_front(
                gamma=gamma,
                symmetric=symmetric,
                batch_size=batch_size,
            )
            t = time.time() - time_zero
            print(f"{name} {batch_size=}", t)
            records.append({"Version": name, "Batch Size": batch_size, "Rep": rep, "Time": t})

            if isinstance(pareto_front, tuple) and len(pareto_front) == 2:
                pareto_rewards, pareto_front = pareto_front

            if isinstance(pareto_front, list):
                pareto_rewards = np.array(pareto_front)
            elif isinstance(pareto_front, pd.DataFrame):
                reward_names = minecart.reward_names
                pareto_rewards = pareto_front[reward_names].values
            else:
                assert isinstance(pareto_front, np.ndarray)

            eq = np.array_equal(pareto_rewards_1, pareto_rewards)
            print(eq)
            assert np.array_equal(pareto_rewards_1, pareto_rewards)

print(f"{frame_skip=}, {gamma=}, {config.name}, {incremental=}, {symmetric=}")
print(f"{len(pareto_front_1)=}")

# Build long-form DataFrame, then pivot to wide
df_long = pd.DataFrame(records)
df = df_long.pivot_table(
    index=["Version", "Batch Size"],
    columns="Rep",
    values="Time",
)
df.columns = [f"Rep {i}" for i in df.columns]
df = df.reset_index()

rep_cols = [c for c in df.columns if c.startswith("Rep ")]
df["Mean"] = df[rep_cols].mean(axis=1)
df["Std"] = df[rep_cols].std(axis=1, ddof=1)
df["log10(Batch Size)"] = np.log10(df["Batch Size"])

print(df)

filename = next(
    f"frameskip{frame_skip}_times({i}).csv"
    for i in range(1, 1000)
    if not Path(f"frameskip{frame_skip}_times({i}).csv").exists()
)

with open(filename, "w") as f:
    f.write(f"# {frame_skip=}, {gamma=}, {config.name}, {incremental=}, {symmetric=}\n")
    f.write(f"# {len(pareto_front)=}\n")
    df.to_csv(f)
# %%
