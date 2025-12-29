import env
import numpy as np


class monte_carlo():
    def __init__(self, env, n_episodes,epsilon, decay_rate, min_epsilon, gamma=0.99):
        self.env = env
        self.n_episodes = n_episodes
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.gamma = gamma
    
    def generate_episode_epsilon(self, policy):
        episode = []
        state = self.env.reset()[0]
        done = False
        while not done:
            action = policy[state]
            if np.random.rand() < self.epsilon:
                action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _= self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated
        return episode
    
    def get_policy_epsilon(self):
        self.Q = np.zeros((self.num_states, self.num_actions))
        sa_total_return = np.zeros((self.num_states, self.num_actions))
        N_sa = np.zeros((self.num_states, self.num_actions))
        policy = np.random.randint(0, self.num_actions, self.num_states)
        for e in range(self.n_episodes):
            episode = self.generate_episode_epsilon(policy)
            rewards = []
            for i in episode:
                rewards.append(i[2])
            visited_states = [] 
            idx = 0
            for step in episode: #state--> step,action,reward
                state = step[0]
                if not state in visited_states:
                    visited_states.append(state)
                    G = 0
                    for t in range(len(rewards[idx:])):
                        G += (self.gamma ** t) * rewards[idx + t]
                    action = step[1]
                    sa_total_return[state,action] += G
                    N_sa[state,action] += 1
                    self.Q[state,action]=sa_total_return[state,action] / (N_sa[state,action] + 1e-10)
                idx += 1
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            policy = np.argmax(self.Q, axis=1)
        return policy

if __name__ == "__main__":
    np.random.seed(1)        
    mc = monte_carlo(env.env, n_episodes=10000, epsilon=1.0, decay_rate=0.995, min_epsilon=0.01)
    policy = mc.get_policy_epsilon()
    print("Optimal Policy: \n{}".format(np.array(policy)))


        
