import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.gridworld import GridWorld
from algorithms.monte_carlo import MonteCarlo

env = GridWorld(size=4)
params = {'gamma': 0.99, 'epsilon': 0.1}
agent = MonteCarlo(env, params, mc_type="first_visit")
results = agent.train(num_episodes=100)
print("Policy:", results['policy'])
print("\n\n")
print("Q (sample):", dict(list(results['Q'].items())[:5]))
print("\n\n")
print("Average reward:", sum(results['episode_rewards']) / len(results['episode_rewards']))

