import gym

from algs.sac_discrete import DiscreteSAC
from parse_args import parse_args
from stable_baselines3.common.logger import configure

args = parse_args()
env = gym.make(args.env_name)

model = DiscreteSAC("MlpPolicy",
    env, verbose=1, learning_rate=args.learning_rate,
    buffer_size=args.buffer_size,
    learning_starts=args.learning_starts,
    batch_size=args.batch_size,
    tau = args.tau,
    gamma=args.gamma,
    train_freq=args.train_freq,
    gradient_steps=args.gradient_steps)
new_logger = configure('./', ["stdout", "csv"])
model.set_logger(new_logger)

model.learn(total_timesteps=args.total_timesteps, 
    log_interval=10)