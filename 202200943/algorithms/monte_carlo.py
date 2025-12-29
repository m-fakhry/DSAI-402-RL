from core.agent import Agent
from .base import epsilon_greedy, greedy
from collections import defaultdict
import random

class MonteCarlo(Agent):
    def __init__(self, env, learning_params, mc_type="first_visit"):
        super().__init__(env, learning_params)
        self.mc_type = mc_type # "First Visit", "Every Visit", "Exploring Starts", "Epsilon Greedy"
        self.episodes = []
        self.Returns_state = defaultdict(list)
        self.Returns_sa = defaultdict(list)
        
        for s in self.env.states:
            if not self.env.is_terminal_state(s):
                random_action = self.env.random_sample_action()
                self.policy[s] = {random_action: 1.0}
    
    def select_action(self, state, training=True):
        if self.env.is_terminal_state(state):
            return self.env.random_sample_action()
        if training:
            return epsilon_greedy(self.epsilon, state, self.Q, self.env)
        else:
            return greedy(state, self.Q, self.env)
        
    def _generate_episode(self):
        episode = []
        state = self.env.reset()
        max_steps = 10000
        
        for _ in range(max_steps):
            action = self.select_action(state, training=True)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            
            if done:
                break
            
            state = next_state
            
        return episode     
    
    def _generate_exploring_starts_episode(self):
        
        non_terminal_states = []
        for s in self.env.states:
            if not self.env.is_terminal_state(s):
                non_terminal_states.append(s)
        
        start_state = random.choice(non_terminal_states)
        start_action = random.choice(self.env.actions)
        self.env.current_state = start_state
        
        episode = []
        next_state, reward, done = self.env.step(start_action)
        episode.append((start_state, start_action, reward))
        
        if done:
            return episode
        
        state = next_state
        max_steps = 10000
        for _ in range(max_steps):
            action = self.select_action(state, training=True)
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            
            if done:
                break
            
            state = next_state
            
        return episode
    
    def _first_visit_mc_update(self, episode):
        
        deltas = []
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            
            G = self.gamma * G + reward
            
            state_visited_before = False
            for i in range(t):
                if episode[i][0] == state:
                    state_visited_before = True
                    break
                
            if not state_visited_before:
                self.Returns_state[state].append(G)
            
            sa_visited_before = False
            
            for i in range(t):
                if episode[i][0] == state and episode[i][1] == action:
                    sa_visited_before = True
                    break
                
            if not sa_visited_before:
                self.Returns_sa[(state, action)].append(G)
            
        for state in self.Returns_state:
            return_list = self.Returns_state[state]
            old_v = self.V[state]
            avg = sum(return_list) / len(return_list)
            self.V[state] = avg
            delta = abs(avg - old_v)
            deltas.append(delta)
        
        for state_action in self.Returns_sa:
            return_list = self.Returns_sa[state_action]
            old_q = self.Q[state_action]
            avg = sum(return_list) / len(return_list)
            self.Q[state_action] = avg
            delta = abs(avg - old_q)
            deltas.append(delta)
                
        if deltas:
            max_delta = max(deltas)
        else:
            max_delta = 0.0
        self.training_data["convergence_metric"].append(max_delta)
                    
                        
    def _every_visit_mc_update(self, episode):
        
        deltas = []
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            
            G = self.gamma * G + reward
            

            self.Returns_state[state].append(G)
        
            self.Returns_sa[(state, action)].append(G)
            
        for state in self.Returns_state:
            return_list = self.Returns_state[state]
            old_v = self.V[state]
            avg = sum(return_list) / len(return_list)
            self.V[state] = avg
            delta = abs(avg - old_v)
            deltas.append(delta)
        
        for state_action in self.Returns_sa:
            return_list = self.Returns_sa[state_action]
            old_q = self.Q[state_action]
            avg = sum(return_list) / len(return_list)
            self.Q[state_action] = avg
            delta = abs(avg - old_q)
            deltas.append(delta)
        
        if deltas:
            max_delta = max(deltas)
        else:
            max_delta = 0.0
        self.training_data["convergence_metric"].append(max_delta)
            
    
    def train(self, num_episodes=1000):
        
        for s in self.env.states:
            self.V[s] = 0.0
            for a in self.env.actions:
                self.Q[(s, a)] = 0.0
                
        for num_episode in range(num_episodes):
            
            if self.mc_type == "exploring_starts":
                episode = self._generate_exploring_starts_episode()
            else:
                episode = self._generate_episode()
            
            self.episodes.append(episode)
            
            if self.mc_type == "every_visit":
                self._every_visit_mc_update(episode)
            else:
                self._first_visit_mc_update(episode)
                
            for s in self.env.states:
                if not self.env.is_terminal_state(s):
                    best_action = self.env.random_sample_action()  
                    max_q = float('-inf')
                    for a in self.env.actions:
                        q_value = self.Q[(s, a)]
                        if q_value > max_q:
                            max_q = q_value
                            best_action = a
                    self.policy[s] = {best_action: 1.0}
            
            episode_reward = 0
            for _, _, r in episode:
                episode_reward += r
            
            episode_length = len(episode)
            
            self.training_data["episode_rewards"].append(episode_reward)                      
            self.training_data["episode_lengths"].append(episode_length)
        
        return {
            "V": dict(self.V),
            "Q": dict(self.Q),
            "policy": dict(self.policy),
            **self.training_data
        }                      