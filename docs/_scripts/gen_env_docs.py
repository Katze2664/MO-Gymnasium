"""
Adapted from https://github.com/Farama-Foundation/Gymnasium/blob/main/docs/scripts/gen_mds.py
"""

import os
import re
from functools import reduce

import gymnasium as gym
import numpy as np
from tqdm import tqdm

import mo_gymnasium as mo_gym


def trim(docstring):
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = 232323
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < 232323:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return "\n".join(trimmed)


pattern = re.compile(r"(?<!^)(?=[A-Z])")

gym.logger.set_level(gym.logger.DISABLED)

all_envs = list(gym.envs.registry.values())
filtered_envs_by_type = {}

# Obtain filtered list
for env_spec in tqdm(all_envs):
    if type(env_spec.entry_point) is not str:
        continue

    split = env_spec.entry_point.split(".")
    # ignore gymnasium.envs.env_type:Env
    env_module = split[0]
    if env_module != "mo_gymnasium":
        continue
    env_type = "environments"  # split[2]
    env_version = env_spec.version

    try:
        env = mo_gym.make(env_spec.id)
        split = str(type(env.unwrapped)).split(".")
        env_name = split[3]

        if env_type not in filtered_envs_by_type.keys():
            filtered_envs_by_type[env_type] = {}
        # only store new entries and higher versions
        if env_name not in filtered_envs_by_type[env_type] or (
            env_name in filtered_envs_by_type[env_type] and env_version > filtered_envs_by_type[env_type][env_name].version
        ):
            filtered_envs_by_type[env_type][env_name] = env_spec

    except Exception as e:
        print(e)

# Sort
filtered_envs = list(
    reduce(
        lambda s, x: s + x,
        map(
            lambda arr: sorted(arr, key=lambda x: x.name),
            map(lambda dic: list(dic.values()), list(filtered_envs_by_type.values())),
        ),
        [],
    )
)

env_dir = os.path.join(os.path.dirname(__file__), "..", "environments")
dir_exists = os.path.exists(env_dir)
if not dir_exists:
    # Create a new directory because it does not exist
    os.makedirs(env_dir)
    print("environments directory has been created!")


# Update Docs
for i, env_spec in tqdm(enumerate(filtered_envs)):
    print("ID:", env_spec.id)
    env_type = env_spec.entry_point.split(".")[2]
    try:
        env = mo_gym.make(env_spec.id)

        # variants dont get their own pages
        e_n = str(env_spec).lower()

        docstring = env.unwrapped.__doc__
        if not docstring:
            docstring = env.unwrapped.__class__.__doc__
        docstring = trim(docstring)

        # pascal case
        pascal_env_name = env_spec.id
        snake_env_name = pattern.sub("_", pascal_env_name).lower()
        # remove what is after the last "-" in snake_env_name e.g. "-v0"
        snake_env_name = snake_env_name[: snake_env_name.rfind("-")]
        title_env_name = snake_env_name.replace("_", " ").title().replace("Mo-", "MO-")
        env_type_title = env_type.replace("_", " ").title()
        related_pages_meta = ""
        if i == 0 or not env_type == filtered_envs[i - 1].entry_point.split(".")[2]:
            related_pages_meta = "firstpage:\n"
        elif i == len(filtered_envs) - 1 or not env_type == filtered_envs[i + 1].entry_point.split(".")[2]:
            related_pages_meta = "lastpage:\n"

        # path for saving video
        v_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "environments",
            # env_type,
            snake_env_name + ".md",
        )

        front_matter = f"""---
autogenerated:
title: {title_env_name}
{related_pages_meta}---
"""
        title = f"# {title_env_name}"
        gif = (
            "```{figure}"
            + f" ../../_static/videos/{env_type}/{snake_env_name}.gif"
            + f" \n:width: 200px\n:name: {snake_env_name}\n```"
        )
        info = (
            "This environment is part of the "
            + f"<a href='..'>{env_type_title} environments</a>."
            + "Please read that page first for general information."
        )
        env_table = "|   |   |\n|---|---|\n"
        env_table += f"| Action Space | {env.action_space} |\n"

        if env.observation_space.shape:
            env_table += f"| Observation Shape | {env.observation_space.shape} |\n"

            if hasattr(env.observation_space, "high"):
                high = env.observation_space.high

                if hasattr(high, "shape"):
                    if len(high.shape) == 3:
                        high = high[0][0][0]
                if env_type == "mujoco":
                    high = high[0]
                high = np.round(high, 2)
                high = str(high).replace("\n", " ")
                env_table += f"| Observation High | {high} |\n"

            if hasattr(env.observation_space, "low"):
                low = env.observation_space.low
                if hasattr(low, "shape"):
                    if len(low.shape) == 3:
                        low = low[0][0][0]
                if env_type == "mujoco":
                    low = low[0]
                low = np.round(low, 2)
                low = str(low).replace("\n", " ")
                env_table += f"| Observation Low | {low} |\n"
        else:
            env_table += f"| Observation Space | {env.observation_space} |\n"

        if env.reward_space.shape:
            env_table += f"| Reward Shape | {env.reward_space.shape} |\n"
        if hasattr(env.reward_space, "high"):
            env_table += f"| Reward High | {env.reward_space.high} |\n"
        if hasattr(env.reward_space, "low"):
            env_table += f"| Reward Low | {env.reward_space.low} |\n"

        env_table += f'| Import | `mo_gymnasium.make("{env_spec.id}")` | \n'

        if docstring is None:
            docstring = "No information provided"
        all_text = f"""{front_matter}
{title}
{env_table}
{docstring}
"""
        file = open(v_path, "w+", encoding="utf-8")
        file.write(all_text)
        file.close()
    except Exception as e:
        print(e)
