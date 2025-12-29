import gymnasium as gym
from core.environment import Environment

class FrozenLake(Environment):
    def __init__(self):
        super().__init__()
        self.gym_env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
        self.actions = list(range(self.gym_env.action_space.n))
        self.num_actions = self.gym_env.action_space.n
        self.states = list(range(self.gym_env.observation_space.n))
        self.terminal_states = [5, 7, 11, 12, 15]  

    def reset(self):
        state, _ = self.gym_env.reset()
        self.current_state = state
        return self.current_state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.gym_env.step(action)
        done = terminated or truncated
        self.current_state = next_state
        return self.current_state, reward, done

    def get_transitions(self, state, action):
        transitions = self.gym_env.unwrapped.P[state][action]
        return [(s_, r, p) for p, s_, r, _ in transitions]

    def render(self):
        print(self.gym_env.render())

    def is_terminal_state(self, state):
        return state in self.terminal_states