# Assignment 1 

The assignment needs to be uploaded to the google classroom no later than 2/27 midnight. 

## Reading 

- Read Chapter 1, Sections 3.1, 3.2, 3.3, 3.4, and 3.5 from the main book

- Read Chapter 1 and 2 from the supplementary book


## Paper-based 

1. A financial analyst is studying a currency exchange rate that changes daily between three states: appreciating, depreciating, or stable. If the exchange rate appreciates on a given day, there is a 25% chance it will continue to appreciate the next day, a 40% chance it will stabilize, and a 35% chance it will depreciate. If the rate depreciates on a particular day, the next day it has a 30% chance to appreciate, 50% chance to depreciate again, and 20% chance to remain stable. When the rate is stable, it is equally likely to appreciate or depreciate the next day.
    - Model this scenario as a Markov chain and write the transition probability matrix

2. A city has four bus stops labeled 1, 2, 3, and 4. Passengers travel between these stops with the following probabilities:
	- From stop 1, passengers go to stop 2 with 35% probability, to stop 3 with 40%, and remain at stop 1 with 25%.
	- From stop 2, 55% travel to stop 1, and 45% go to stop 4.
	- From stop 3, 20% go to stop 2, 60% stay at stop 3, and 20% go to stop 4.
	- From stop 4, 30% travel to stop 1, 30% to stop 2, 20% to stop 3, and 20% remain at stop 4.

    - Perform the following tasks
        - Write the transition matrix T
        - What is the probability that a passenger currently at stop 3 will travel to stop 4 next?
    	- What is the probability that a passenger currently at stop 3 will reach stop 4 after two stops?
    	- What is the probability that a passenger currently at stop 3 will stay at stop 3 and not travel to any other stop next?


## Implementation 

- **Markov Process**
  - Define a transition probability matrix 
  - Start from a chosen initial state, simulate the Markov chain for a large number of transitions 
  - Record the state at each step of the simulation
  - Use the library [networkx](https://networkx.org/) to plot the transition matrix as a graph 

- **Markov Decision Process**

	As shown in Figure 3.2 in the main book, the figures illustrates a simple MDP represented by a rectangular gridworld. Each cell in the grid corresponds to a state in the environment. From any cell, the agent can take one of four possible actions: move `north`, `south`, `east`, or `west`. These actions **deterministically** move the agent one cell in the chosen direction. If an action would cause the agent to move off the grid, its position remains the same, but it receives a reward of -1. All other moves yield a reward of 0, except for actions taken in two special states, A and B. From state A, any action gives a reward of 10 and moves the agent to state A’. Similarly, from state B, any action yields a reward of 5 and moves the agent to state B’. Write a python program that performs the following
  1. Define a function to return which action to apply for a given state (policy $\pi(a|s)$)
  2. Define a function to return the next state and reward given the current state and action $p(s^\prime | s,a)$ and $r(s,a,s^\prime)$
  3. Start from a chosen initial state, simulate the Markov chain for a large number of transitions (1000 steps)
  4. Repeat the above step multiple times to generate as many processes as you can 
  5. Compute the value function for each state under two different reward functions (Assuming $\gamma=.9$)
      1. reward = -1 if the agent moves off the grid, reward = 10 for any action from state A, reward = 5 for any action from state B, reward = 0 for all cases
      2. reward = 5 if the agent moves off the grid, reward = 16 for any action from state A, reward = 11 for any action from state B, reward = 6 for all cases
  6. Compare and discuss how the value functions change under these two reward settings
