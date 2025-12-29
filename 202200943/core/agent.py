from abc import ABC, abstractmethod
from collections import defaultdict

class Agent(ABC):
    
    def __init__(self, env, learning_params):
        # Learning Parameters
        self.env = env
        self.gamma = learning_params.get('gamma', 0.99)
        self.alpha = learning_params.get('alpha', 0.1)
        self.epsilon = learning_params.get('epsilon', 0.1)
        
        # Value and Policy Storage
        self.V = defaultdict(float) # State-Value Function
        self.Q = defaultdict(float) # Action-Value Function
        self.policy = {} # State -> Action List
        
        self.training_data = {
            'episode_rewards': [],
            'episode_lengths': [],
            'convergence_metric': []
        }
        
    @abstractmethod
    def train(self, num_episodes):
        """
        This is the complete training loop for the agent,
        the agent interacts with the environment for N 
        episodes or iterations. in each episode, the agent observe states,
        select actions, recieve rewards, and update its learning parametes.
        """
        pass
    
    @abstractmethod
    def select_action(self, state, training=True):
        """
        The agent uses its policy to choose an action for
        the given state.
        """
        pass
    
    def get_state_value(self):
        """
        Return current state-value function V(s)
        state-value function is the expected return starting
        from state s and following policy pi.
        """
        return dict(self.V)
    
    def get_action_value(self):
        """
        Return current Q-function Q(s,a)
        Q-function or action value function means that starting 
        in state s and taking action a then you start following 
        policy pi so, the first action you take is not under policy pi
        but, from the second action now you choose it under policy pi
        """
        return dict(self.Q)
    
    def get_policy(self):
        """Return current policy"""
        return dict(self.policy)
    
    def run_episode(self, training=True, max_steps=1000):
        """
        Run one episode using current policy.
        
        Returns:
            Episode: List of (state, action, reward) tuples
            states_visited: List of states in order
            total_reward: Sum of rewards
        """
        episode = []
        states_visited = []
        total_reward = 0.0
        
        state = self.env.reset()
        states_visited.append(state)
        
        for _ in range(max_steps):
            action = self.select_action(state, training=training)
            next_state, reward, done = self.env.step(action)
            
            episode.append((state, action, reward))
            total_reward += reward
            states_visited.append(next_state)
            
            if done:
                break
            
            state = next_state
        
        return episode, states_visited, total_reward
    
    def reset(self):
        """Reset all learned values"""
        self.V.clear()
        self.Q.clear()
        self.policy.clear()
        self.training_data = {
            'episode_rewards': [],
            'episode_lengths': [],
            'convergence_metric': []
        }