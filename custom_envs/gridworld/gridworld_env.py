# This file is taken from https://github.com/Alfo5123/Robust-Multitask-RL/blob/master/code/envs/gridworld_env.py
# with some modifications 

import gym
import sys
import os
import copy
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import seeding
from collections import OrderedDict
from numpy.random import randint


EMPTY = BLACK = 0
WALL = GRAY = 1
TARGET = GREEN = 3
AGENT = RED = 4
SUCCESS = PINK = 6
COLORS = {BLACK: [0.0, 0.0, 0.0], GRAY: [0.5, 0.5, 0.5], GREEN: [0.0, 1.0, 0.0],
          RED: [1.0, 0.0, 0.0], PINK: [1.0, 0.0, 1.0]}

NOOP = 4
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 0


class GridworldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    num_env = 0

    def __init__(self, plan, generate_goal=False, random_start=False):
        super().__init__()
        self.plan = plan

        self.max_step = 10000000
        self.actions = [NOOP, UP, DOWN, LEFT, RIGHT]
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(4)  # Only consider 4 action, exclude NOOP
        self.action_pos_dict = {NOOP: [0, 0], UP: [-1, 0], DOWN: [1, 0], LEFT: [0, -1], RIGHT: [0, 1]}

        self.img_shape = [800, 800, 3]  # visualize state

        # initialize system state
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, 'plan{}.txt'.format(plan))
        self.start_grid_map = self._read_grid_map(self.grid_map_path)  # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.grid_map_shape = self.start_grid_map.shape
        self.observation_space = spaces.Box(low=np.array([0., 0.]),
                                            high=np.array([1.0, 1.0]))

        # agent state: start, target, current state
        self.agent_start_state, self.agent_target_state = self._get_agent_start_target_state()
        if random_start:
            self.start_grid_map[self.agent_start_state] = EMPTY
            self.agent_start_state = None
        
        self.agent_state = copy.deepcopy(self.agent_start_state)
        
        if generate_goal:
            self.generate_task()
        
        if self.agent_state is None:
            self.agent_state = self._place_agent()

        # set other parameters
        self.restart_once_done = False  # restart or not once done

        # set seed
        self.seed()

        # consider total episode reward
        self.episode_total_reward = 0.0

        # consider viewer for compatibility with gym
        self.viewer = None

        # counter for time step
        self.time = 0

        self.penalty_step = 0.1
        self.penalty_wall = 0.5
        

        self.optimal_reward = None
        # calculate the length of shortest path from start to goal
        # self.shortest_path_length = self._find_shortest_path_length_start_goal()
        # print('Length of shortest path:', self.shortest_path_length)

    def seed(self, seed=None):

        # Fix seed for reproducibility

        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def generate_task(self):
        n_trial = 1000
        while n_trial >= 0:
            state = randint(0, self.grid_map_shape[0]), randint(0, self.grid_map_shape[1])
            
            if self.start_grid_map[state] != WALL:
                self.start_grid_map[self.agent_target_state] = EMPTY
                self.agent_target_state = state
                self.start_grid_map[self.agent_target_state] = TARGET
                return state
            n_trial -= 1
        print('Cannot find a valid starting position!')
        return None

    def get_state(self, coordinates, action, reward):

        # Return a tuple with: current location of the agent in the map
        # given coordinates

        # Normalized for better perform of the NN

        return np.asarray([coordinates[0] / self.grid_map_shape[0],
                           coordinates[1] / self.grid_map_shape[1]])

    def step(self, action):
        # Wrapper for step method, used to terminate the episode when
        # number of time steps exceed the horizon
        next_state, reward, done, info = self._step(action)
        if self.time >= self.max_step:
            done = True
        self.time += 1
        if done:
            info['rollout_return'] = self.episode_total_reward
        return next_state, reward, done, info

    def _step(self, action):

        # Return next observation, reward, finished, success

        action = int(action)
        info = {'success': False}
        done = False

        # Penalties
        penalty_step = self.penalty_step
        penalty_wall = self.penalty_wall

        reward = -penalty_step
        nxt_agent_state = (self.agent_state[0] + self.action_pos_dict[action][0],
                           self.agent_state[1] + self.action_pos_dict[action][1])

        if action == NOOP:
            info['success'] = True
            self.episode_total_reward += reward  # Update total reward
            return self.get_state(self.agent_state, action, reward), reward, False, info

        # Make a step
        next_state_out_of_map = (nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]) or \
                                (nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1])

        if next_state_out_of_map:
            info['success'] = False
            self.episode_total_reward += reward  # Update total reward
            return self.get_state(self.agent_state, action, reward), reward, False, info

        # successful behavior
        target_position = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]

        if target_position == EMPTY:

            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = AGENT

        elif target_position == WALL:

            info['success'] = False
            self.episode_total_reward += (reward - penalty_wall)  # Update total reward
            return self.get_state(self.agent_state, action, reward - penalty_wall), (reward - penalty_wall), False, info

        elif target_position == TARGET:

            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = SUCCESS

        self.current_grid_map[self.agent_state[0], self.agent_state[1]] = EMPTY
        self.agent_state = copy.deepcopy(nxt_agent_state)
        info['success'] = True

        if nxt_agent_state[0] == self.agent_target_state[0] and nxt_agent_state[1] == self.agent_target_state[1]:
            done = True
            # The final reward is added with the peripheral of the map, 
            # multiplied by the step penalty, this step will ensure that
            # the maximum rewards achievable of all environments configuration 
            # are scaled suitably with the size of the map
            reward += self.get_reward_on_reaching_goal()
            if self.restart_once_done:
                self.reset()

        self.episode_total_reward += reward  # Update total reward
        return self.get_state(self.agent_state, action, reward), reward, done, info

    def reset(self):

        # Return the initial state of the environment

        self.agent_state = copy.deepcopy(self.agent_start_state)
        if self.agent_state is None:  # Random start
            self.agent_state = self._place_agent()
            # self.shortest_path_length = self._find_shortest_path_length_start_goal()

        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.current_grid_map[self.agent_state] = AGENT
        self.episode_total_reward = 0.0
        self.time = 0
        return self.get_state(self.agent_state, 0.0, 0.0)

    def close(self):
        if self.viewer: self.viewer.close()

    def _read_grid_map(self, grid_map_path):

        # Return the gridmap imported from a txt plan

        grid_map = open(grid_map_path, 'r').readlines()
        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array, dtype=int)
        return grid_map_array

    def _get_agent_start_target_state(self):
        start_state = np.where(self.start_grid_map == AGENT)
        target_state = np.where(self.start_grid_map == TARGET)

        if not target_state[0]:
            sys.exit('Target state not specified')

        target_state = (target_state[0][0], target_state[1][0])

        if not start_state[0]:
            # print('Start state not found, generate random start at each episode')
            return None, target_state
        start_state = (start_state[0][0], start_state[1][0])

        return start_state, target_state

    def _gridmap_to_image(self, img_shape=None):

        # Return image from the gridmap

        if img_shape is None:
            img_shape = self.img_shape
        observation = np.zeros(img_shape, dtype=float)
        gs0 = int(observation.shape[0] / self.current_grid_map.shape[0])
        gs1 = int(observation.shape[1] / self.current_grid_map.shape[1])
        for i in range(self.current_grid_map.shape[0]):
            for j in range(self.current_grid_map.shape[1]):
                for k in range(3):
                    this_value = COLORS[self.current_grid_map[i, j]][k]
                    observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1, k] = this_value
        return (255 * observation).astype(np.uint8)

    def render(self, mode='human', close=False):
        """
        Returns a visualization of the environment according to specification
        """

        if close:
            plt.close(1)  # Final plot
            return

        img = self._gridmap_to_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            plt.figure()
            plt.imshow(img)
            return

    def get_reward_on_reaching_goal(self):
        return 1.0 + self.penalty_step * np.sum(self.grid_map_shape) * 2

    def get_optimal_reward(self):
        """
        Get the expected optimal reward
        """
        ret = []
        for state in self._get_all__empty_coord():
            shortest = self._find_shortest_path_length_start_goal(state)
            rw = self.get_reward_on_reaching_goal() - shortest * self.penalty_step
            if shortest < 0:
                ret.append(-self.penalty_step * (self.max_step + 1))
            else:
                ret.append(rw)

        return ret

    def _get_all__empty_coord(self):
        for y in range(self.grid_map_shape[0]):
            for x in range(self.grid_map_shape[1]):
                coord = y, x
                if self.start_grid_map[coord] == EMPTY:
                    yield coord

    def _place_agent(self):
        '''
        Randomly find an empty space to place agent
        '''
        n_trial = 1000
        while n_trial >= 0:
            from numpy.random import randint
            state = randint(0, self.grid_map_shape[0]), randint(0, self.grid_map_shape[1])
            if self.start_grid_map[state] == EMPTY:
                return state
            n_trial -= 1
        print('Cannot find a valid starting position!')
        return None

    def _additional_check_start_position(self, plan: int, state: np.ndarray):
        '''
        For some specific environments, there are some additional requirements
        for the starting place of the agent, this function will do that.
        '''
        if plan == 20:
            return state[1] < 7
        elif plan == 21:
            return state[1] > 7
        return True

    def _check_valid(self, coord):
        """
        Given a coordinate, return True if this coordinate is inside the map
        or not a wall, i.e it is a valid coordinate that agent can take
        """
        next_state_out_of_map = (coord[0] < 0 or coord[0] >= self.grid_map_shape[0]) or \
                                (coord[1] < 0 or coord[1] >= self.grid_map_shape[1])

        if next_state_out_of_map: return False
        if self.current_grid_map[coord[0], coord[1]] == WALL: return False
        return True

    def _find_shortest_path_length_start_goal(self, start=None):
        """
        Return the length of the shortest path from start to goal,
        this function is used to assign reward for reaching goal
        since when the map is too large and the path to goal is too long,
        the total return will be dominated by step penalty,
        thus, the final reward should be scale appropriately with the length
        of the path.
        This method uses BFS to search for the shortest path
        If there are no such paths, return -1
        Params:
            start: starting position, if None, the starting position
            will be chosen to be the agent starting position
        """
        from queue import Queue
        if start is None:
            start = copy.deepcopy(self.agent_state)

        q = Queue()
        q.put(start)

        visit = np.zeros(self.grid_map_shape, dtype=bool)
        visit[start] = True

        shortest_length = 0
        actions = np.array(list(self.action_pos_dict.values()), dtype=int)

        while not q.empty():
            for _ in range(q.qsize()):
                node = q.get()

                if node == self.agent_target_state:
                    return shortest_length

                for action in actions:
                    neighbor_node = tuple(action + node)

                    if not self._check_valid(neighbor_node): continue
                    if visit[neighbor_node]: continue

                    visit[neighbor_node] = True;
                    q.put(neighbor_node)

            shortest_length += 1
        return -1

from gym import GoalEnv

class MultiGoalGridWorld(GoalEnv):
    """
    Wrapper class for Gridworld environment, with multi-goal settup
    this class provides compatibility with HER 
    """
    def __init__(self, plan):
        super().__init__()
        self.env = GridworldEnv(plan)
        self.action_space = self.env.action_space
        self.observation_space = spaces.Dict(
            {
                "observation": copy.deepcopy(self.env.observation_space),
                "achieved_goal": copy.deepcopy(self.env.observation_space),
                "desired_goal": copy.deepcopy(self.env.observation_space),
            }
        )
        
    def seed(self, seed: int) -> None:
        self.env.seed(seed)
    
    def _get_obs(self, current_state, desired_goal):
        return OrderedDict(
            [
                ("observation", copy.deepcopy(current_state)),
                ("achieved_goal", copy.deepcopy(current_state)),
                ("desired_goal", copy.deepcopy(desired_goal)),
            ]
        )
    
    def step(self, action: int):
        next_state, reward, done, info = self.env.step(action)
        desired_goal = self.env.get_state(self.env.agent_target_state, action, reward)
        obs = self._get_obs(next_state, desired_goal)
        return obs, reward, done, info
        
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    def reset(self):
        state = self.env.reset()
        
        return self._get_obs(state, self.env.get_state(self.env.agent_target_state, None, None))
        
    def compute_reward(self, achieved_goal, desired_goal, _info):
        if np.abs(achieved_goal - desired_goal).sum() < 1e-8: return self.env.get_reward_on_reaching_goal()
        else: return -self.env.penalty_step