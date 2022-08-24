import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training SAC continuous')

    parser.add_argument('--env_name', default='LunarLander-v2')

    parser.add_argument('--buffer_size', default=1000000, type=int)

    parser.add_argument('--learning_starts', default=1000, type=int)
    parser.add_argument('--total_timesteps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--gradient_steps', default=1, type=int)
    parser.add_argument('--train_freq', default=1, type=int)
   
    parser.add_argument('--eval_interval', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
   
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
   
    parser.add_argument('--num_layers', default=3, type=int)

    parser.add_argument('--gamma', default=0.99, type=float)

    # parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')

    args, unknown = parser.parse_known_args()
    return args

