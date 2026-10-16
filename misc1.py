# Checks modified minecarts produce the same pareto fronts and convex sets as the original (mc1)
# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd

import mo_gymnasium.envs.minecart.minecart1 as mc_a

# import mo_gymnasium.envs.minecart.minecart2 as mc_b
# import mo_gymnasium.envs.minecart.minecart3 as mc_b
# import mo_gymnasium.envs.minecart.minecart4 as mc_b
# import mo_gymnasium.envs.minecart.minecart5 as mc_b
# import mo_gymnasium.envs.minecart.minecart6 as mc_b
# import mo_gymnasium.envs.minecart.minecart7 as mc_b
# import mo_gymnasium.envs.minecart.minecart8 as mc_b
import mo_gymnasium.envs.minecart.minecart_b1 as mc_b

config_stochastic = Path(r"C:\Users\ethan\Desktop\Cloned repos\MO-Gymnasium\mo_gymnasium\envs\minecart\mine_config.json")
config_det = Path(r"C:\Users\ethan\Desktop\Cloned repos\MO-Gymnasium\mo_gymnasium\envs\minecart\mine_config_det.json")

frame_skip = 4
batch_size = 10**2

for gamma in [0.98, 1]:
    for config in [config_det, config_stochastic]:
        for incremental in [True, False]:
            for symmetric in [True, False]:
                print(f"{frame_skip=}, {gamma=}, {config.name}, {incremental=}, {symmetric=}")
                minecart_a = mc_a.Minecart(
                    frame_skip=frame_skip,
                    incremental_frane_skip=incremental,
                    config=config,
                )
                minecart_b = mc_b.Minecart(
                    frame_skip=frame_skip,
                    incremental_frane_skip=incremental,
                    config=config,
                )
                pareto_front_a = minecart_a.pareto_front(
                    gamma=gamma,
                    symmetric=symmetric,
                )
                pareto_front_b = minecart_b.pareto_front(
                    gamma=gamma,
                    symmetric=symmetric,
                    batch_size=batch_size,
                )
                convex_a = minecart_a.convex_coverage_set(
                    gamma=gamma,
                    symmetric=symmetric,
                )
                convex_b = minecart_b.convex_coverage_set(
                    gamma=gamma,
                    symmetric=symmetric,
                    batch_size=batch_size,  # mc4 to mc8
                )

                # mc1
                assert isinstance(pareto_front_a, list)
                assert isinstance(convex_a, list)
                pareto_rewards_a = np.array(pareto_front_a)
                convex_rewards_a = np.array(convex_a)

                # mc2 to mc4, mc_b1
                assert isinstance(pareto_front_b, list)
                assert isinstance(convex_b, list)
                pareto_rewards_b = np.array(pareto_front_b)
                convex_rewards_b = np.array(convex_b)

                # mc5
                # assert isinstance(pareto_front_b, np.ndarray)
                # assert isinstance(convex_b, np.ndarray)
                # pareto_rewards_b = pareto_front_b
                # convex_rewards_b = convex_b

                # mc6
                # assert isinstance(pareto_front_b, pd.DataFrame)
                # reward_names = minecart_b.reward_names
                # pareto_rewards_b = pareto_front_b[reward_names].values
                # convex_rewards_b = convex_b[reward_names].values

                # mc7 to mc8
                # assert isinstance(pareto_front_b, tuple)
                # assert len(pareto_front_b) == 2
                # assert isinstance(pareto_front_b[0], np.ndarray)
                # assert isinstance(pareto_front_b[1], pd.DataFrame)
                # assert isinstance(convex_b, tuple)
                # assert len(convex_b) == 2
                # assert isinstance(convex_b[0], np.ndarray)
                # assert isinstance(convex_b[1], pd.DataFrame)
                # pareto_rewards_b = pareto_front_b[0]
                # convex_rewards_b = convex_b[0]

                pareto_eq = np.array_equal(pareto_rewards_a, pareto_rewards_b)
                # convex_eq = np.array_equal(convex_rewards_a, convex_rewards_b)
                print(pareto_eq)
                # print(convex_eq, len(convex_rewards_a), len(convex_rewards_b))

                assert np.array_equal(pareto_rewards_a, pareto_rewards_b)
                # assert np.array_equal(convex_rewards_a, convex_rewards_b)  # mc2 to mc7

# %%
