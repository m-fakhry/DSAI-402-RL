from abc import ABC, abstractmethod

class Environment(ABC):
    
    def __init__(self):
        self.states = []               
        self.actions = []              
        self.current_state = None      
        self.terminal_states = []      
        self.num_actions = 0
    
    @abstractmethod
    def reset(self):
        """
        Reset the Environment.
        """
        pass
    
    @abstractmethod
    def step(self, action):
        """
        Execute one step in the environment.
        """
        pass
    
    @abstractmethod
    def get_transitions(self, state, action):
        """
        Provides all possible next states, rewards, and their probabilities for a state-action pair.
        """
        pass
    
    @abstractmethod
    def render(self) -> None:
        """
        Displays the current state of the environment.
        """
        pass
    
    def random_sample_action(self):
        import random
        return random.choice(self.actions)
    
    def is_terminal_state(self, state):
        """Check if state is terminal"""
        return state in self.terminal_states
    
