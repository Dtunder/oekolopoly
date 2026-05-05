"""Plots training curves for agents"""

import seaborn as sns
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from tqdm import tqdm
from utils import decode_from_agent_string
from agent_lists import PPO_FLUC

WINDOW = 501
POST = "_smoothed"
AGENTS = PPO_FLUC
PLOT_LIST = ['balance', 'round', 'r']
PL = sns.color_palette("colorblind")

fig, axs = plt.subplots(1, len(PLOT_LIST), figsize=(10, 3.33), sharex=True)

for j, agent in enumerate(pbar := tqdm(AGENTS)):
    pbar.set_description(f"Plotting {agent}")
    obs, act, rew, shape, drl, seed = decode_from_agent_string(agent)
    df = pd.read_csv(f"./agents/{agent}/train.monitor.csv", header=1)

    df["timestep"] = df["l"].cumsum()

    for i, col in enumerate(PLOT_LIST):
        df[col + POST] = savgol_filter(df[col], WINDOW, 5)
        sns.lineplot(x=df["timestep"],
                     y=df[col + POST],
                     ax=axs[i],
                     c=PL[j],
                     label=str(seed),
                     lw=1)

# Plot cosmetics
for i in range(len(PLOT_LIST)):
    axs[i].get_legend().remove()
    axs[i].ticklabel_format(axis='x', style='sci', scilimits=(5, 5),
                            useMathText=True)

axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].set_ylabel(r"balance $B$")
axs[1].set_ylabel(r"rounds $r$")
axs[2].set_ylabel("return")

handles, labels = axs[0].get_legend_handles_labels()
custom_lines = [Line2D([0], [0], color=PL[i], lw=6) for i in range(len(AGENTS))]
fig.legend(handles=custom_lines, labels=labels, ncol=5, title="Seed",
           loc="center", bbox_to_anchor=(0.5, -0.1))
plt.subplots_adjust(wspace=0.25)
plt.savefig("./training_deviations.pdf", bbox_inches="tight")
plt.close()
