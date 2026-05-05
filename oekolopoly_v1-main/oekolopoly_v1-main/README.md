This repository contains code for the paper ["Ökolopoly: Case Study on Large Action Spaces in Reinforcement Learning"](https://doi.org/10.1007/978-3-031-53966-4_9) presented in 2023 at the conference "Machine Learning, Optimization, and Data Science" (LOD 2023).

# Description of Directories and Files
- `oekolopoly` directory:
  - `wrappers.py` contains the wrappers to modify observation- and actionspace as well as reward functions
  - `oekolopoly` contains the RL environment
  - `oekolopoly_gui` the necessary files to run the GUI for human-play
  - `my_oekolopoly_gui` contains the files to run the GUI 2.0 for human-play
- `agent_list.py` lists of agents used for different investigations
- `trained agents` contains a few trained PPO agents with high performance
- `evaluate.py` evaluates the given list of agents by computing mean and standard deviation of the last evaluation episodes during training. The results are saved in a csv-file specified in the last line of code
- `train.py` trains an agent with the parameters passed to the script (wrappers, DRL algorithm, parameters for reward-shaping etc.)
- `utils.py` a few functions for comodity
- `additional material` contains plots which due to the page limit could not be part of the paper. These include all pie chrts for episodes' termination (see Fig. 6 in [the publication](#Publication)) and fluctuations in the training curves for PPO, SAC, and TD3 algorithms (see Fig. 2 in [the publication](#Publication)).

# How to...
## ...train an agent?
To train a PPO agent without observation wrapper, using Simple action wrapper and PerRound reward wrapper with a constant per-round reward of $R_c=0.5$, and seed 17 one would issue the command:

	python3 train.py --observation "none" --action "simple" --reward "perround" --shaping 0.5 --algo "ppo" --seed 17
## ...play using the GUI?
	python3 -m oekolopoly.oekolopoly_gui.oeko_gui
## ...use the environment in my own scripts?
Once you import via

    import oekolopoly.oekolopoly

the environment is registered and available just like any other environment known from OpenAI Gym

    env = gym.make("Oekolopoly-v1")

# Requirements
The experiments were performed on Python version 3.10.7
The complete list of packages as per `pip freeze` are:
```
absl-py==1.4.0
ale-py==0.7.4
AutoROM==0.4.2
AutoROM.accept-rom-license==0.5.0
cachetools==5.2.1
certifi==2022.12.7
charset-normalizer==3.0.1
click==8.1.3
cloudpickle==2.2.0
commonmark==0.9.1
contourpy==1.0.6
cycler==0.11.0
fonttools==4.38.0
google-auth==2.16.0
google-auth-oauthlib==0.4.6
grpcio==1.51.1
gym==0.21.0
idna==3.4
importlib-metadata==4.13.0
importlib-resources==5.10.2
joblib==1.2.0
kiwisolver==1.4.4
libtorrent==2.0.7
Markdown==3.4.1
MarkupSafe==2.1.1
matplotlib==3.6.3
numpy==1.24.1
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
oauthlib==3.2.2
opencv-python==4.7.0.68
packaging==23.0
pandas==1.5.2
Pillow==9.4.0
protobuf==3.20.3
psutil==5.9.4
pyasn1==0.4.8
pyasn1-modules==0.2.8
pygame==2.1.2
Pygments==2.14.0
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2022.7
requests==2.28.2
requests-oauthlib==1.3.1
rich==13.0.1
rsa==4.9
sb3-contrib==1.7.0
scikit-learn==1.2.1
scipy==1.10.0
seaborn==0.12.2
six==1.16.0
stable-baselines3==1.7.0
tensorboard==2.11.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
threadpoolctl==3.1.0
torch==1.13.1
tqdm==4.64.1
typing_extensions==4.4.0
urllib3==1.26.14
Werkzeug==2.2.2
zipp==3.11.0
```

# Publication
To reference the paper:
```
@InProceedings{10.1007/978-3-031-53966-4_9,
author     = "Engelhardt, Raphael C.
             and Raycheva, Ralitsa
             and Lange, Moritz
             and Wiskott, Laurenz
             and Konen, Wolfgang",
editor     = "Nicosia, Giuseppe
             and Ojha, Varun
             and La Malfa, Emanuele
             and La Malfa, Gabriele
             and Pardalos, Panos M.
             and Umeton, Renato",
title      = "{\"O}kolopoly: Case Study on Large Action Spaces in Reinforcement Learning",
booktitle  = "Machine Learning, Optimization, and Data Science",
bookseries = "Lecture Notes in Computer Science",
volume     = "14506",
year       = "2024",
publisher  = "Springer Nature Switzerland",
address    = "Cham",
pages      = "109--123",
isbn       = "978-3-031-53966-4",
doi        = "10.1007/978-3-031-53966-4\_9"
}
```
