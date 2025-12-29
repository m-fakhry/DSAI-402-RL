import numpy as np
import random

def epsilon_greedy(epsilon, state, q_values, env):
    """
    The epsilon greedy is an action-selection strategy that is designed 
    to balance exploration(trying new things to discover better options
    and not get stuck) and exploitation(using what the agent already
    knows to maximize rewards)
    it flips a coin if the result is less than epsilon it takes random action
    from the environment (exploration) else it loops through all the actions and
    build a list of their Q values(how good each action in that state)
    and select the action with highest q value (exploitation) 
    """
    if random.random() < epsilon:
        return env.random_sample_action()
    else:
        actions = env.actions
        q_vals = []
        for action in actions:
            key = (state, action)
            value = q_values.get(key, 0.0)
            q_vals.append(value)
            
        return actions[np.argmax(q_vals)]    
    
def greedy(state, q_values, env):
    
    actions = env.actions
    q_vals = []
    for action in actions:
        key = (state, action)
        value = q_values.get(key, 0.0)
        q_vals.append(value)
    return actions[np.argmax(q_vals)]    