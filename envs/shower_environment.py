from gymnasium import Env
from gymnasium.spaces import Discrete
import numpy as np
import random

class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Discrete(101)
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60

    def step(self, action):
        self.state += action - 1
        self.state = np.clip(self.state, 0, 100)

        self.shower_length -= 1

        reward = 1 if 37 <= self.state <= 39 else -1
        terminated = False
        truncated = self.shower_length <= 0
        return np.array([self.state]), reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60
        return self.state, {}
    
    def render(self):
        pass