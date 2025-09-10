import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Tuple
import numpy as np
from watchtower.utils import get_delta_func


class CMFEnv(gym.Env):
    def __init__(self, initial_trajectory: tuple[int, ...], delta_func: callable):
        '''
        A custom environment for optimizing a trajectory to maximize the delta of a given CMF.

        Parameters:
        initial_trajectory (tuple): The initial trajectory to start from.
        delta_func (callable): A function that computes the delta of the CMF given a trajectory.
        '''
        super(CMFEnv, self).__init__()

        self.initial_trajectory = initial_trajectory
        self.delta_func = delta_func

        # Define action and observation space
        self.action_space = Discrete(len(initial_trajectory) * 2)
        self.observation_space = Tuple((Box(low=0, high=100, shape=(len(initial_trajectory),), dtype=np.int32), Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)))

        self.state = np.array(initial_trajectory, dtype=np.int32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.array(self.initial_trajectory, dtype=np.int32)
        delta = self.delta_func(tuple(self.state))

        return self.state, {'delta': delta}

    def step(self, action: int):
        done = False

        var_index = action // 2
        change = 1 if action % 2 == 0 else -1

        new_state = self.state.copy()
        new_state[var_index] += change
        new_delta = self.delta_func(tuple(new_state))

        # stay in place
        if new_delta == -np.inf:
            reward = -1.0
        else:
            reward = new_delta - self.delta_func(tuple(self.state))
            self.state = new_state

        info = {'delta': self.delta_func(tuple(self.state))}

        return self.state, reward, done, False, info


if __name__ == '__main__':
    from ramanujantools.cmf.known_cmfs import pi
    import sympy as sp

    cmf = pi()
    constant = 'pi'

    x, y = sp.symbols('x, y')
    variables = (x, y)
    starting_point = {x: 1, y: 1}

    cost_fn = get_delta_func(cmf, starting_point, constant, 5000, variables)

    initial_trajectory = (1, 1)
    env = CMFEnv(initial_trajectory, cost_fn)

    state, info = env.reset()
    print(f"Initial state: {state}, delta: {info['delta']}")

    for _ in range(10):
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        print(f"Action: {action}, New state: {state}, Reward: {reward}, Delta: {info['delta']}")
