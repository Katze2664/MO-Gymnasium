# %%
import time

t0 = time.time()

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

import mo_gymnasium.envs.minecart.minecart_b1_len as mc


def summarise(raw_record):
    return {
        "frame_skip": raw_record["frame_skip"],
        "trimmed_sequence_sum": sum(raw_record["trimmed_sequence_lens"]),
        "seq_counters_sum": sum(raw_record["seq_counters"]),
        "max_len": max(raw_record["max_lens"]),
        "max_pareto_rewards": max(raw_record["pareto_rewards_lens"]),
        "final_pareto_rewards": raw_record["pareto_rewards_lens"][-1],
        "trimmed_sequence_time_sum": sum(raw_record["trimmed_sequence_times"]),
        "seq_counter_time_sum": sum(raw_record["seq_counter_times"]),
        "batch_filter_time_sum": sum(raw_record["batch_filter_times"]),
        "final_filter_time": raw_record["final_filter_time"],
        "total_time": raw_record["total_time"],
    }


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_dir = Path(".") / "results" / timestamp
os.makedirs(run_dir, exist_ok=False)

config_stochastic = Path(mc.__file__).parent / "mine_config.json"
config_det = Path(mc.__file__).parent / "mine_config_det.json"

gamma = 0.98
config = config_det
assert config.exists(), config
incremental = True
symmetric = False
batch_size = 10**5
frame_skips = [4, 3, 2, 1]

print(f"{gamma=}, {config.name}, {incremental=}, {symmetric=}, {batch_size=}")

raw_records = []
records = []
for frame_skip in frame_skips:
    print(f"{frame_skip=}")
    t1 = time.time()
    minecart = mc.Minecart(
        frame_skip=frame_skip,
        incremental_frame_skip=incremental,
        config=config,
    )
    pareto_front, raw_data = minecart.pareto_front(
        gamma=gamma,
        symmetric=symmetric,
        batch_size=batch_size,
    )
    raw_data["frame_skip"] = frame_skip
    raw_records.append(raw_data)

    # Save raw data first, so a bug in summarise() can't lose the computation
    with open(run_dir / f"raw_records{frame_skip}.pkl", "wb") as file:
        pickle.dump(raw_records, file)

    # Rebuild the CSV after every frame skip, so a timeout keeps all completed rows
    records.append(summarise(raw_data))
    pd.DataFrame(records).to_csv(run_dir / "frame_skip_records.csv", index=False)

    print(f"Time for {frame_skip=}: {time.time() - t1}\n")

with open(run_dir / "raw_records.pkl", "wb") as file:
    pickle.dump(raw_records, file)

df = pd.DataFrame(records)
print(df)
for record in records:
    for k, v in record.items():
        print(k, v)

print(f"Total time: {time.time() - t0}")

# %%
