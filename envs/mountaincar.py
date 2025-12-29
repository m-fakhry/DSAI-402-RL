import gymnasium as gym
import numpy as np
from core.environment import Environment

class MountainCar(Environment):
    def __init__(self, bins_per_dim=20):
        super().__init__()
        self.gym_env = gym.make("MountainCar-v0")
        self.bins_per_dim = bins_per_dim

        self.low = self.gym_env.observation_space.low * 1.1  
        self.high = self.gym_env.observation_space.high * 1.1

        self.bins = [
            np.linspace(self.low[i], self.high[i], bins_per_dim + 1)
            for i in range(2)
        ]

        self.actions = [0, 1, 2]
        self.num_actions = 3

        self.states = list(range(bins_per_dim ** 2))

    def _discretize(self, obs):
        indices = []
        for i in range(2):
            idx = np.digitize(obs[i], self.bins[i]) - 1
            idx = np.clip(idx, 0, self.bins_per_dim - 1)
            indices.append(idx)
        return indices[0] * self.bins_per_dim + indices[1]

    def reset(self):
        obs, _ = self.gym_env.reset()
        self.current_state = self._discretize(obs)
        return self.current_state

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.gym_env.step(action)
        done = terminated or truncated
        self.current_state = self._discretize(obs)
        return self.current_state, reward, done

    def get_transitions(self, state, action):
        raise NotImplementedError("No analytical transition model")

    def render(self):
        print(f"Discrete state ID: {self.current_state}")

    def is_terminal_state(self, state):
        return False