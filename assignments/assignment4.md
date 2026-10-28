# Assignment 4

Not Graded.

## Reading

- Read Sections 4.1 to 4.4 of Sutton and Barto (policy evaluation, policy improvement, policy iteration and value iteration).
- Read Section 5.1 (Monte Carlo prediction).

## Implementation: Blackjack

In this assignment a single problem (Blackjack) is solved with policy iteration, value iteration and Monte Carlo. The three methods are then compared. Policy iteration and value iteration cannot run without the model $p(s', r \mid s, a)$. Monte Carlo only needs to play games. Section 5.1 of the book says that DP is hard to use for Blackjack because the model is hard to work out. You will test this claim.

### The game

Create the Blackjack environment of Gymnasium with `sab=False`.

```python
import gymnasium as gym
env = gym.make("Blackjack-v1", sab=False)
```

These are the rules the environment follows.

- An ace counts as 1 or 11. J, Q and K count as 10.
- The deck never runs out, i.e. every card is drawn with replacement. Each value from 1 to 9 comes up with probability $1/13$ and the value 10 comes up with probability $4/13$.
- You get two cards. The dealer also gets two cards but shows only one of them.
- There are two actions: 0 = stick (stop) and 1 = hit (take one more card).
- If your sum goes above 21 you lose right away and the reward is $-1$.
- Once you stick, the dealer keeps taking cards until the dealer's sum reaches 17 or more. If the dealer goes above 21 you win. If not, the higher sum wins. A win gives $+1$, a draw gives $0$ and a loss gives $-1$.
- An ace is usable if it can count as 11 and the sum stays at 21 or below.
- The state is (your sum, the dealer's card, usable ace) and $\gamma = 1$.

With `sab=False`, an ace and a ten as your first two cards count as an ordinary 21. This way the state above has everything you need to predict what happens next. The default `sab=True` breaks this. Can you see why?

### Part A. Write the model

You write the model first because DP cannot run without it. Write down how long this part takes you since you need this number in Part E.

1. How many non-terminal states are there? Count the sums below 12 as well because the environment gives them too. Explain how you got your number.
2. For a hit, find the next state (or a bust) for each state and each card value.
3. For a stick, the reward depends on where the dealer ends. For each dealer card from ace to 10, compute the probability that the dealer ends on 17, 18, 19, 20 or 21 or goes above 21. A short recursive function over the dealer's hand is enough. Make a table with the probability that the dealer goes above 21 for each dealer card.
4. Store the model in the same format that Gymnasium uses for FrozenLake, i.e. `P[s][a]` is a list of `(prob, next_state, reward, done)` tuples.

### Part B. Policy iteration and value iteration

1. Implement iterative policy evaluation (in place, with threshold $\theta = 10^{-9}$), policy improvement and policy iteration as in the lecture. Start from the policy "always stick". When two actions have the same value, keep the current action.
2. Implement value iteration with the same threshold.
3. For policy iteration, report the number of iterations, how many states change their action in each iteration and the total number of evaluation sweeps. For value iteration, report the number of sweeps. Give the run time of both methods.
4. Do the two methods find the same policy? Report the largest gap between their value functions.
5. Draw the optimal policy as two grids. One grid is for the states with a usable ace and the other is for the states without one. Each grid shows your sum (12 to 21) against the dealer's card (ace to 10). Compare your grids with Figure 5.2 in the book.
6. Run value iteration two times. In the first run, sweep the states from low sums to high sums. In the second run, sweep them from high sums to low sums. How many sweeps does each order need and why? Hint: can a hand ever return to a state it was in before?

### Part C. Monte Carlo prediction

Evaluate the policy that sticks only on 20 or 21. Example 5.1 in the book uses the same policy.

1. Implement first-visit MC prediction, which learns from episodes that you play in the environment.
2. Find the true $v_\pi$ of this policy with your policy evaluation from Part B.
3. Run MC for 1,000, 10,000, 100,000 and 500,000 episodes. For each run, report the root mean square error (RMSE) between the MC values and the true values over the 200 states with sums 12 to 21. A state that MC never visits keeps its starting value of 0. Plot the error against the number of episodes on log-log axes. By how much does the error go down when you play 10 times more episodes?
4. Which states have the largest error? Why?
5. In Blackjack, every-visit MC gives the same answer as first-visit MC. Why?

With 500,000 episodes, MC and DP should be close on most states. If they are not, your model or your MC code has a bug.


### Part D. Compare the three methods

1. Fill in this table.

| | Policy iteration | Value iteration | MC prediction |
|---|---|---|---|
| Needs the model? | | | |
| What it gives you | | | |
| Number of iterations, sweeps or episodes | | | |
| Run time | | | |
| Your time to write the code (add Part A for DP) | | | |
| Error of the values against the exact ones | | | |

2. Use your policy evaluation to find the expected return per game of the optimal policy and of three simple policies, which are "always stick", "stick only on 20 or 21" and "copy the dealer" (stick on 17 or more). Average the values over the start states. Hint: the start state comes from three cards (two for you and one for the dealer) so you can list all $10 \times 10 \times 10$ card combinations with their probabilities. Then estimate the same four numbers with MC, i.e. play 100,000 games with each policy and average the rewards. Do the DP and MC numbers agree?
3. Answer each question in a few lines.
   - To find the value of a policy in Blackjack, would you pick DP or MC prediction? Why?
   - Was the book right that DP is hard to use for Blackjack?
   - In Example 5.3 the book compares its MC policy with Thorp's basic strategy. The two differ in one place for a usable ace. The book is not sure why but says it is confident that its policy is optimal for its version of the game. Does your DP agree with the book?
   - Think about your class project. Can you write $p(s', r \mid s, a)$ for your environment? If you cannot, which of the three methods can you still use?


