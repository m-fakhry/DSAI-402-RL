import gymnasium as gym

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
env2= gym.make("Taxi-v3", render_mode="human")
env3= gym.make("CliffWalking-v1", render_mode="human")
env4= gym.make("Blackjack-v1", render_mode="human")
env5= gym.make("CartPole-v1", render_mode="human")
env6= gym.make("MountainCar-v0", render_mode="human")

