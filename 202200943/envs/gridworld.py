from core.environment import Environment
import random


class GridWorld(Environment):
    
    def __init__(self, size=4, start=(0, 0), goal=None):
        super().__init__()
        
        self.size = size
        self.start = start
        self.goal = goal if goal is not None else (size - 1, size - 1)
        self.current_state = self.start
        
        # Create all possible states
        self.states = [(i, j) for i in range(size) for j in range(size)]
        
        self.actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
        self.action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.num_actions = 4        
        self.terminal_states = [self.goal]
    
    def reset(self):
        self.current_state = self.start
        return self.current_state
    
    def step(self, action):
        x, y = self.current_state
        
        if action == 0:  # UP
            x = max(0, x - 1)
        elif action == 1:  # DOWN
            x = min(self.size - 1, x + 1)
        elif action == 2:  # LEFT
            y = max(0, y - 1)
        elif action == 3:  # RIGHT
            y = min(self.size - 1, y + 1)
        
        self.current_state = (x, y)
        
        reward = -1.0
        
        done = self.current_state in self.terminal_states
        
        return self.current_state, reward, done
    
    def get_transitions(self, state, action):
        x, y = state
        
        if action == 0:  # UP
            next_x = max(0, x - 1)
            next_y = y
        elif action == 1:  # DOWN
            next_x = min(self.size - 1, x + 1)
            next_y = y
        elif action == 2:  # LEFT
            next_x = x
            next_y = max(0, y - 1)
        elif action == 3:  # RIGHT
            next_x = x
            next_y = min(self.size - 1, y + 1)
        
        next_state = (next_x, next_y)
        reward = -1.0
        probability = 1.0  # Deterministic
        
        return [(next_state, reward, probability)]
    
    def render(self):
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        # Mark agent position
        ax, ay = self.current_state
        grid[ax][ay] = 'A'
        
        # Mark goal position
        gx, gy = self.goal
        grid[gx][gy] = 'G'
        
        # Print grid
        print("\n" + "=" * (self.size * 2 + 1))
        for row in grid:
            print(' '.join(row))
        print("=" * (self.size * 2 + 1) + "\n")
    
    def random_sample_action(self):
        return random.choice(self.actions)