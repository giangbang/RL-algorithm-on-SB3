# SAC Discrete Stable-baselines3

Set of implementations of RL algorithms, using Stable-baselines3, 

## Installation

```
git clone https://github.com/giangbang/RL-algorithms-on-SB3.git
cd RL-algorithms-on-SB3
pip install -r requirements.txt
```

## How to run

Discrete SAC
```
python train.py --env_name LunarLander-v2 --gradient_steps 1 --train_freq 1 --learning_rate 3e-4
```

Distral
```
python train_multitask.py
```