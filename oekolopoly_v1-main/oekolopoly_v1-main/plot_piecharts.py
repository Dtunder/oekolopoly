import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

#import agent_lists

def parse_done_reasons(string):
    tmp = string.split(".")[0]
    return tmp

def plot_piechart(data, ax):
    # Prepare dataset for done reasons
    sel = data.iloc[-WINDOW:]
    sel["done_reason"] = sel["done_reason"].map(parse_done_reasons)
    count = sel.done_reason.value_counts()
    count = count.to_frame()
    count = count.reset_index()
    count.loc[count['done_reason'] < 0.05*sum(count.done_reason), 'index'] = 'Other'
    count = count.groupby('index')['done_reason'].sum().reset_index()
    count.set_index("index", inplace=True)

    count.plot.pie(y="done_reason", ax=ax, autopct='%.0f%%', ylabel='', legend=True, labeldistance=None)

    ax.set_ylabel('')
    ax.legend(bbox_to_anchor=(0.5, 0), loc='upper center')

WINDOW = 1000
import os
AGENTS = [f.name for f in os.scandir("../agents") if f.is_dir()]
# raise
# AGENTS = ["obs_box_action_box_reward_perround_0.5_ppo_17",
#           "obs_none_action_simple_reward_perround_1.0_ppo_17",
#           "obs_simple_action_box_reward_perround_0.75_ppo_17"]

# fig, ax = plt.subplots(1, 3, figsize=(15, 5*int(np.ceil(len(AGENTS)/3))))

for i, agent_str in enumerate(pbar := tqdm(AGENTS)):
    fig, ax = plt.subplots(1, 1)
    pbar.set_description(f"Plotting {agent_str}")
    df = pd.read_csv(f"../agents/{agent_str}/eval.monitor.csv", header=1)

    plot_piechart(df, ax)
    ax.set_title(f"{agent_str}", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{agent_str}.png", bbox_inches="tight")
    plt.close()

#plt.suptitle("Reasons for episode termination")
#plt.show()


