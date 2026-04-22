# Assignment 7

**Objective:** To understand the fundamental differences between SARSA and Q-Learning through manual calculation and Python implementation, specifically focusing on how each algorithm handles risk and exploration.


## Paper-based 


Consider a 1D grid world with three states: $S_1$ (Safe), $S_2$ (Risk), and $S_{Swamp}$ (Terminal). There are two actions: $A_{stay}$ and $A_{move}$. 
- At $S_1$: Taking $A_{move}$ transitions the agent to $S_2$ with Reward $R = -1$. $A_{stay}$ results in the agent remaining in $S_1$ with Reward $R = -1$.
- At $S_2$: The action $A_{stay}$ let the agent remain in $S_2$ with Reward $R = -1$. The action $A_{move}$ let the agent transition to $S_{Swamp}$ with Reward $R = -100$.

An agent is navigating the 1D grid world. It is currently at state $S_1$ and takes action $A_{move}$ to arrive at state $S_2$. Assume $\gamma=0.5$, $\alpha=0.5$, $Q(S_1, A_{move}) = 0$, $Q(S_2, A_{stay}) = -2$ and $Q(S_2, A_{move}) = -100$.

**Tasks:**

1. Q-Learning Update: Calculate the updated value for $Q(S_1, A_{move})$ using the Q-Learning rule
2. SARSA Update: Assume the agent is following an $\epsilon$-greedy policy. Upon arriving at $S_2$, the policy chooses to explore and selects the next action $A_{t+1} = A_{move}$ (falling into the swamp). Calculate the updated value for $Q(S_1, A_{move})$ using the SARSA rule
3. Analysis: Compare the two results. How does SARSA's update reflect the "danger" of $S_2$ even though the agent hasn't actually taken the move into the swamp yet?. i.e. Even though the agent hasn't fallen into the swamp yet, why is SARSA's updated value so much lower (more negative) than Q-Learning's? What does this tell you about SARSA's "awareness" of its own exploration?

## Implementation 

Using Python and the `gymnasium` library, implement both SARSA and Q-Learning agents to solve the 1D grid world environment

1. Implementation: Complete the update logic for both algorithms in a single training script
2. Parameters: Use $\epsilon = 0.1$, $\alpha = 0.5$, and $\gamma = 0.9$
3. Visualization: Plot the Cumulative Reward per Episode for both algorithms on the same graph. Print the final learned policy (the path) for each agent

### Starter Code Snippet:
```python
import gymnasium as gym
import numpy as np

env = gym.make('CliffWalking-v0')

def update_q_learning(q_table, s, a, r, s_next, alpha, gamma):
    # TODO: Implement Q-Learning update
    pass

def update_sarsa(q_table, s, a, r, s_next, a_next, alpha, gamma):
    # TODO: Implement SARSA update
    pass
```


**Submission: Evaluation & Report**

1. The Path Difference: Describe the final paths. Which algorithm learns to walk along the edge, and which one takes the "long way" around?
2. Online Performance: Why does SARSA often show a higher total reward during the training phase compared to Q-Learning in this environment?
3. Real-World Application: If you were training a robot that could physically break if it falls, which algorithm is more appropriate to use during the learning process?


<!-- 
Paper-based (40%):
- Manual Calculations 25% 
- Analysis & Logic 15% 
Implementation (60%): 
- Code Implementation 30% 
- Visualization 10% 
- Evaluation & Report 20%  
-->
