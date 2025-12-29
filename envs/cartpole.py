import gymnasium as gym
import numpy as np
from core.environment import Environment

class CartPole(Environment):
    def __init__(self, bins_per_dim=10):
        super().__init__()
        self.gym_env = gym.make("CartPole-v1")
        self.bins_per_dim = bins_per_dim


        self.low = self.gym_env.observation_space.low * 1.1 
        self.high = self.gym_env.observation_space.high * 1.1
        self.low[1] = -5.0  
        self.low[3] = -5.0
        self.high[1] = 5.0
        self.high[3] = 5.0

        self.bins = [
            np.linspace(self.low[i], self.high[i], bins_per_dim + 1)
            for i in range(4)
        ]

        self.actions = [0, 1]
        self.num_actions = 2

        self.states = list(range(bins_per_dim ** 4))
    
    def _discretize(self, obs):
        indices = []
        for i in range(4):
            idx = np.digitize(obs[i], self.bins[i]) - 1
            idx = np.clip(idx, 0, self.bins_per_dim - 1)
            indices.append(idx)
        state = 0
        for i, idx in enumerate(indices):
            state += idx * (self.bins_per_dim ** i)
        return int(state)

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
        raise NotImplementedError("No model available for continuous env")

    def render(self):
        print(f"Discrete state: {self.current_state}")

    def is_terminal_state(self, state):
        return False