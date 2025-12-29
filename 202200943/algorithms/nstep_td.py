from core.agent import Agent
from .base import epsilon_greedy, greedy

class NstepTemporalDifference(Agent):
    def __init__(self, env, learning_params):
        super().__init__(env, learning_params)
        self.n = learning_params.get('n', 3)  
        for s in self.env.states:
            if not self.env.is_terminal_state(s):
                random_action = self.env.random_sample_action()
                self.policy[s] = {random_action: 1.0}
                
    
    def select_action(self, state, training = True):
        if self.env.is_terminal_state(state):
            return self.env.random_sample_action()
        if training:
            return epsilon_greedy(self.epsilon, state, self.Q, self.env)
        else:
            return greedy(state, self.Q, self.env)
    
    def train(self, num_episodes=1000):
        for s in self.env.states:
            for a in self.env.actions:
                self.Q[(s, a)] = 0.0
        
        for num_episode in range(num_episodes):
            
            S = {}
            A = {}
            R = {}
            
            state = self.env.reset()
            S[0 % (self.n + 1)] = state # Exactly saying S[0] = state
            A[0 % (self.n + 1)] = self.select_action(state, training=True)
            T = float("inf")
            episode_reward = 0.0
            episode_length = 0
            max_steps = 10000
            max_delta = 0.0
            
            t = 0
            while True:
                if t < T:
                    action = A[t % (self.n + 1)]
                    next_state, reward, done = self.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    
                    R[(t + 1) % (self.n + 1)] = reward
                    S[(t + 1) % (self.n + 1)] = next_state
                    
                    if done:
                        T = t + 1
                    else:
                        next_action = self.select_action(next_state, training=True)
                        A[(t + 1) % (self.n + 1)] = next_action
                
                tau = t - self.n + 1
                if tau >= 0:
                    
                    G = 0.0
                    
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        reward_i = R[i % (self.n + 1)]
                        power = i - tau - 1
                        G += (self.gamma ** power) * reward_i
                        
                    if tau + self.n < T:
                        next_n_state = S[(tau + self.n) % (self.n + 1)]
                        next_n_action = A[(tau + self.n) % (self.n + 1)]
                        G += (self.gamma ** self.n) * self.Q[(next_n_state, next_n_action)]
                        
                    state_tau = S[tau % (self.n + 1)]
                    action_tau = A[tau % (self.n + 1)]
                    old_q = self.Q[(state_tau, action_tau)]
                    self.Q[(state_tau, action_tau)] += self.alpha * (G - old_q)
                    delta = abs(self.Q[(state_tau, action_tau)] - old_q)
                    if delta > max_delta:
                        max_delta = delta
                
                if tau >= T - 1:
                    break
                
                t += 1
                
                if t > max_steps:
                    break
            
            self.training_data["episode_rewards"].append(episode_reward)
            self.training_data["episode_lengths"].append(episode_length)
            self.training_data["convergence_metric"].append(max_delta)
            
            for s in self.env.states:
                if not self.env.is_terminal_state(s):
                    best_action = self.env.random_sample_action()  
                    max_q = float('-inf')
                    for a in self.env.actions:
                        q_value = self.Q.get((s, a), 0.0)
                        if q_value > max_q:
                            max_q = q_value
                            best_action = a
                    self.policy[s] = {best_action: 1.0}
            
        return {
            "Q": dict(self.Q),
            "policy": dict(self.policy),
            **self.training_data
        }                                              