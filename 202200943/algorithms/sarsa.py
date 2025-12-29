from core.agent import Agent
from .base import epsilon_greedy

class Sarsa(Agent):
    def __init__(self, env, learning_params):
        super().__init__(env, learning_params)
        for s in self.env.states:
            if not self.env.is_terminal_state(s):
                random_action = self.env.random_sample_action()
                self.policy[s] = {random_action: 1.0}
                
    def select_action(self, state, training = True):
        if self.env.is_terminal_state(state):
            return self.env.random_sample_action()

        return epsilon_greedy(self.epsilon, state, self.Q, self.env)
    
    def train(self, num_episodes=1000):
        
        for s in self.env.states:
            for a in self.env.actions:
                self.Q[(s, a)] = 0.0
                
        for num_episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            max_steps = 1000
            max_delta = 0.0
            
            action = self.select_action(state, training=True)
            
            for _ in range(max_steps):
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                next_action = self.select_action(next_state, training=True)
                q_next = self.Q[(next_state, next_action)]
                old_q = self.Q[(state, action)]
                td_error = reward + self.gamma * q_next - old_q
                self.Q[(state, action)] += self.alpha * td_error
                
                delta = abs(self.Q[(state, action)] - old_q)
                if delta > max_delta:
                    max_delta = delta
                    
                if done:
                    break
                
                state = next_state
                action = next_action
                
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