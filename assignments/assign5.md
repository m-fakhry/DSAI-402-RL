# Assignment 5

**Learning How to Play Atari Breakout**

Atari Breakout is a classic arcade game where the player controls a paddle at the bottom of the screen to bounce a ball and destroy a wall of bricks at the top; check this [youtube video](https://www.youtube.com/watch?v=tT70Tv6D41o). The objective is to eliminate all bricks by hitting them with the ball, which bounces off the paddle and walls. If the ball passes the paddle and leaves the screen, the player loses a turn, and the game ends when all turns are used up or all bricks are destroyed. Each brick destroyed earns points, and the game becomes progressively harder as the ball speeds up and the paddle may shrink. The Atari [Breakout dataset](https://github.com/takuseno/d4rl-atari) includes the following components:

- **States**: Each state is a grayscale image of the game screen, resized to 84x84 pixels.
- **Actions**: Discrete actions: move paddle left, move paddle right, or do nothing.
- **Rewards**: +1 for each brick destroyed, 0 for no change, -1 if the ball is lost.
- **Terminals**: Boolean flag indicating episode end (all bricks destroyed or all lives lost).

**Tasks:**

1. **Counterexample**: Provide a scenario where value iteration fails to converge in an MDP if the discount factor is set to 1. Explain why this happens and its implications for Breakout.
2. **Scenario**: Construct a simple MDP where a greedy policy performs worse than an epsilon-greedy policy in Breakout. Justify your answer with a scenario.
3. **Implementation**:
   - Implement and compare policy evaluation using Monte Carlo first visit and every visit.
   - Implement value iteration and policy iteration for a simplified state space.
   - Use epsilon-greedy exploration in a Monte Carlo control setting.
   - Analyze and compare results in terms of convergence speed, final value estimates, and policy performance.
