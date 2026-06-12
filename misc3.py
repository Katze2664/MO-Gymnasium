
# Plots time taken to run different pareto front versions
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

path = Path(f"frameskip2_times(1).csv")
df = pd.read_csv(path, comment="#")

for key, grp in df.groupby("Version"):
    # print(grp)
    plt.errorbar(
        x=grp["log10(Batch Size)"],
        y=grp["Mean"], label=key,
        yerr=grp["Std"],
    )
plt.legend()
# plt.ylim(top=600)
plt.show()

# %%
