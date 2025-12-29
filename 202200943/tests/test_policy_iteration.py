import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.gridworld import GridWorld
from algorithms.policy_iteration import PolicyIteration

env = GridWorld(size=4)
params = {'gamma': 0.99, 'theta': 1e-6}
agent = PolicyIteration(env, params)
results = agent.train()
print("Policy:", results['policy'])
print("\n\n")
print("V:", results['V'])
print("\n\n")
print("Convergence iterations:", results['convergence_metric'][-1])