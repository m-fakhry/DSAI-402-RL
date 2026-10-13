---
theme: neversink
layout: cover
colorSchema: light
routerMode: hash
transition: slide-left
title: RL (DSAI 402)
author: Mohamed Ghalwash
year: Fall 2026-2027
venue: Zewail City
class: 'text-center'
mdc: true
lecture: 4- DP
slide:
  disableSlideNumbers: true
slide_info: false
---

# Reinforcement Learning <br> (DSAI 402)
## Lecture 4: Bellman Equations and Policy Iteration

**Prof. Mohamed Ghalwash**
<Email v="mghalwash@zewailcity.edu.eg" />
_Zewail City University_

:: note ::

Lecture 4, Monday 12 October 2026

---
layout: top-title
color: sky-light
align: lt
title: Today
---

:: title ::

# Agenda

:: content ::

**Objectives**: you should be able to write the Bellman equations for $v_\pi$ and $q_\pi$, compute the value of a policy by repeated sweeps, improve a policy from its values, and run policy iteration on a small MDP.

- Play: the class becomes the grid and every square holds a card and an arrow
- Name it: backup, policy evaluation, policy improvement, policy iteration
- Formalize it: the Bellman equations, then policy evaluation, policy improvement and policy iteration
- Apply it: the third application paper, the AI Clinician for sepsis

<!--
Timing: about 25 minutes for the game, 35 for the Bellman equations, 40 for policy iteration, and 15 for the paper.
If time is tight, shorten the convergence slide and the common mistakes slide.
-->

---
layout: top-title
color: light
align: lt
title: Play first
---

:: title ::

# The Grid Is Back and You Are the Squares

:: content ::

<div class="ns-c-tight">

- Same map as last week, but the floor is **not slippery**. Every move goes where the arrow points.
- Same points: $-1$ for any move, $-5$ more for landing on 💥, and $+10$ more for landing on ⭐. Hitting a wall keeps you in place and costs $-1$.
- 11 volunteers stand in the shape of the grid. Each one is a square and holds a **card** that starts at 0 and the **arrow** shown below. The ⭐ card is always 0.

</div>

| | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| **Row 1** | → 0 | → 0 | → 0 | ↓ 0 |
| **Row 2** | 🟢 → 0 | 💥 → 0 | → 0 | ⭐ |
| **Row 3** | → 0 | → 0 | 💥 → 0 | ↑ 0 |

<!--
If there is no room for 11 volunteers, every team does the same thing on paper and one team reads its grid aloud after each round.
The starting arrows are a lazy plan: go right until the last column, then go to the star. Two of the paths walk onto a bomb.
-->

---
layout: top-title
color: light
align: lt
title: The first rule
---

:: title ::

# Card Rounds

:: content ::

- In a **card round**, every square writes a new card: the points of the move its arrow makes + the card of the square it lands on.
- Use the cards **from the last round**. Everyone shows the new card at the same time.
- Keep doing card rounds until a round changes no card.

| Square | Round | Arrow lands on | Points | Card there, last round | New card |
|---|---|---|---|---|---|
| 🟢 (2,1) | 1 | 💥 (2,2) | $-1 - 5 = -6$ | 0 | $-6$ |
| (1,3) | 2 | (1,4) | $-1$ | 9 | $8$ |

<AdmonitionType type='tip'>
A square only talks to its neighbors. Nobody looks at the whole map.
</AdmonitionType>

<!--
Make sure everyone uses the cards of the last round, not the ones being written right now. That is why all cards are shown at the same time.
-->

---
layout: top-title
color: light
align: lt
title: Card round 1
---

:: title ::

# Card Round 1

:: content ::

Only the squares one move away from the ⭐, along their arrows, know about it.

| | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| **Row 1** | → -1 | → -1 | → -1 | ↓ 9 |
| **Row 2** | 🟢 → -6 | 💥 → -1 | → 9 | ⭐ |
| **Row 3** | → -1 | → -6 | 💥 → -1 | ↑ 9 |

---
layout: top-title
color: light
align: lt
title: Card round 2
---

:: title ::

# Card Round 2

:: content ::

The squares two moves away now know about the ⭐. The others still only see costs.

| | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| **Row 1** | → -2 | → -2 | → 8 | ↓ 9 |
| **Row 2** | 🟢 → -7 | 💥 → 8 | → 9 | ⭐ |
| **Row 3** | → -7 | → -7 | 💥 → 8 | ↑ 9 |

---
layout: top-title
color: light
align: lt
title: Card round 3
---

:: title ::

# Card Round 3

:: content ::

The squares three moves away now know about the ⭐. The arrow of 🟢 leads through the 💥 so its card is $-6 + 8 = 2$.

| | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| **Row 1** | → -3 | → 7 | → 8 | ↓ 9 |
| **Row 2** | 🟢 → 2 | 💥 → 8 | → 9 | ⭐ |
| **Row 3** | → -8 | → 2 | 💥 → 8 | ↑ 9 |

---
layout: top-title
color: light
align: lt
title: Card round 4
---

:: title ::

# Card Round 4

:: content ::

Only (1,1) and (3,1) change. They are the farthest from the ⭐ along their arrows.

| | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| **Row 1** | → 6 | → 7 | → 8 | ↓ 9 |
| **Row 2** | 🟢 → 2 | 💥 → 8 | → 9 | ⭐ |
| **Row 3** | → 1 | → 2 | 💥 → 8 | ↑ 9 |

Round 5 gives the same cards so we stop. Each card is now the total points the robot collects from that square if it follows the arrows.

---
layout: top-title
color: light
align: lt
title: The second rule
---

:: title ::

# Arrow Rounds

:: content ::

- When a card round changes nothing, do an **arrow round**. Every square tries all 4 moves with the current cards and switches only if another move gives a **strictly larger** sum.
- After an arrow round, go back to card rounds.

Here is 🟢 after card round 4.

| Move | ↑ | → | ↓ | ← |
|---|---|---|---|---|
| Lands on | (1,1) | 💥 (2,2) | (3,1) | wall, stays |
| Points + card there | $-1 + 6$ | $-6 + 8$ | $-1 + 1$ | $-1 + 2$ |
| Sum | **5** | 2 | 0 | 1 |

↑ gives 5, which beats the 2 of its own arrow. So 🟢 switches to ↑.

---
layout: top-title
color: light
align: lt
title: Can we do better
---

:: title ::

# Who Switched?

:: content ::

| Arrow round | Who switches | Card before | Card after | Why |
|---|---|---|---|---|
| 1 | 🟢 (2,1): → to ↑ | 2 | 5 | Going over the top avoids the 💥 |
| 2 | (3,1): → to ↑ | 1 | 4 | It can now follow the new path of 🟢 |
| 3 | (3,2): → to ← | 2 | 3 | The long way around beats the 💥 |
| 4 | nobody | | | We stop |

- After each switch, one card round updates the cards and the next card round changes nothing.
- Each switch makes the squares behind it better, which gives the next square a reason to switch.

<!--
Ties do happen. In arrow round 2, (3,2) gets 2 from → and also 2 from ↑. The strict rule keeps its arrow so the game cannot go around in circles.
-->

---
layout: top-title
color: light
align: lt
title: The end of the game
---

:: title ::

# The Final Arrows

:: content ::

| | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| **Row 1** | → 6 | → 7 | → 8 | ↓ 9 |
| **Row 2** | 🟢 ↑ 5 | 💥 → 8 | → 9 | ⭐ |
| **Row 3** | ↑ 4 | ← 3 | 💥 → 8 | ↑ 9 |

No square wants to switch. Later today we see why these arrows are the best possible.

---
layout: top-title
color: light
align: lt
title: Reflect
---

:: title ::

# Debrief

:: content ::

- Did any square ever see the whole map? <span v-click="1" style="background: #fff3b0; padding: 0.1em 0.5em; border-radius: 0.3em;">no, each square only used its neighbors' cards</span>

- What did a card mean when the card rounds stopped? <span v-click="2" style="background: #fff3b0; padding: 0.1em 0.5em; border-radius: 0.3em;">the total points from that square if the robot follows the current arrows</span>

- Why did a square switch its arrow? <span v-click="3" style="background: #fff3b0; padding: 0.1em 0.5em; border-radius: 0.3em;">taking another move once and then following the current arrows was worth more than its own arrow</span>

- Did any card ever go down after a switch? <span v-click="4" style="background: #fff3b0; padding: 0.1em 0.5em; border-radius: 0.3em;">no. Switching to a better move never makes things worse</span>

- How did we know when to stop? <span v-click="5" style="background: #fff3b0; padding: 0.1em 0.5em; border-radius: 0.3em;">an arrow round changed no arrow</span>

- Could you do this without knowing the points and where each move lands? <span v-click="6" style="background: #fff3b0; padding: 0.1em 0.5em; border-radius: 0.3em;">no. Every square needed the full rules of the game, i.e. the model</span>

---
layout: grid-cards
cols: 3
---

# You Already Used All of These

:: card-1 ::
#### Backup
Update the value of a state from the values of the states it can reach.
- Game: writing a new card from a neighbor's card

:: card-2 ::
#### Policy evaluation
Repeat backups for a fixed policy until the values stop changing.
- Game: the card rounds

:: card-3 ::
#### Policy improvement
In every state, switch to the action with the best one-step look ahead.
- Game: an arrow round

:: card-4 ::
#### Policy iteration
Evaluate, improve, and repeat until the policy stops changing.
- Game: the whole thing you just did

:: card-5 ::
#### Bellman equation
The rule behind every card round. A value is the reward now plus the discounted value of where you land.
- Game: points + the neighbor's card

:: card-6 ::
<AdmonitionType type='important'>
The game had no dice so it was easy. Today we write the same rules with dice, i.e. with an average over what the world might do.
</AdmonitionType>

---
layout: top-title
color: light
align: lt
title: Recap
---

:: title ::

# Recap: Return and Value

:: content ::

The return is the discounted sum of rewards from time $t$ on. It has a recursive form.

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = R_{t+1} + \gamma G_{t+1}
$$

The value of a state is the return we expect if we start there and follow $\pi$.

$$
v_\pi(s) = \mathbb{E}_\pi\left[ G_t \mid S_t = s \right]
$$

The value of an action is the return we expect if we take $a$ now and follow $\pi$ after that.

$$
q_\pi(s,a) = \mathbb{E}_\pi\left[ G_t \mid S_t = s,\ A_t = a \right]
\qquad
v_\pi(s) = \sum_a \pi(a \mid s)\, q_\pi(s,a)
$$

<AdmonitionType type='important'>
A return is what happened in one run. A value is what you expect on average.
</AdmonitionType>

---
layout: top-title
color: light
align: lt
title: Value from the neighbors
---

:: title ::

# The Bellman Equation for $v_\pi$

:: content ::

Start from the definition and use $G_t = R_{t+1} + \gamma G_{t+1}$.

$$
\begin{aligned}
v_\pi(s) &= \mathbb{E}_\pi\left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s \right] \\
&= \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma\, \mathbb{E}_\pi\left[ G_{t+1} \mid S_{t+1} = s' \right] \right] \\
&= \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma\, v_\pi(s') \right]
\end{aligned}
$$

- The second line averages over the action we pick and over what the world does next. It uses the Markov property, i.e. the return after $s'$ does not depend on how we got to $s'$.
- The third line is the definition of $v_\pi(s')$.

In words, the value of a state is the average, over what we might do and what the world might do, of the reward now plus the discounted value of where we land.

---
layout: top-title
color: light
align: lt
title: Reading the equation
---

:: title ::

# Reading the Bellman Equation

:: content ::

| Part | What it does | In the game |
|---|---|---|
| $\sum_a \pi(a \mid s)$ | Average over our actions | One arrow, so one term |
| $\sum_{s', r} p(s', r \mid s, a)$ | Average over what the world does | No dice, so one term |
| $r + \gamma\, v_\pi(s')$ | Reward now + discounted next value | Points + neighbor's card, $\gamma = 1$ |

For 🟢 with its first arrow →, both sums have a single term, i.e.

$$
v_\pi(2,1) = (-1 - 5) + v_\pi(2,2) = -6 + 8 = 2
$$

That is its card after round 4.

For a fixed policy we get one linear equation for each state. With $n$ states there are $n$ equations and $n$ unknowns.

---
layout: top-title
color: emerald-light
align: lt
title: With dice
---

:: title ::

# The Same Equation on the Slippery Grid

:: content ::

Last week the floor was slippery. Take the top right square under last week's best arrows. Its arrow is ↓ and $\gamma = 1$.

- Rolls 1 to 4 (4 out of 6): the robot lands on the ⭐ and gets $-1 + 10 = 9$.
- Roll 5 (1 out of 6): the move turns to →, a wall stops the robot, and it pays $1$ and stays.
- Roll 6 (1 out of 6): the move turns to ←, and the robot pays $1$ and lands one square to the left, which is worth $6.6$.

$$
v = \tfrac{4}{6}(9) + \tfrac{1}{6}(-1 + v) + \tfrac{1}{6}(-1 + 6.6) \quad\Rightarrow\quad v \approx 8.1
$$

<AdmonitionType type='tip'>
This is the Bellman equation for one state. The policy picks one arrow and the dice give three outcomes, so the sum over the next state has three terms.
</AdmonitionType>

<!--(5/6) v = 6 - 1/6 + 5.6/6, so v is about 8.1.-->

---
layout: top-title
color: light
align: lt
title: Value of an action
---

:: title ::

# The Bellman Equation for $q_\pi$

:: content ::

$$
q_\pi(s,a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \sum_{a'} \pi(a' \mid s')\, q_\pi(s', a') \right]
$$

Both equations say the same thing. The value of a choice is the reward we get now plus the discounted value of what comes next.

We can also write $q_\pi$ from $v_\pi$. We will use this form for policy improvement.

$$
q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma\, v_\pi(s') \right]
$$

<AdmonitionType type='tip'>
In the arrow rounds, each of the 4 sums a square tried was the value of one action, i.e. q for that move.
</AdmonitionType>

---
layout: top-title
color: light
title: A small example
---

:: title ::

# Solving for $v_\pi$ by Hand

:: content ::

Two states $A$ and $B$, with $\gamma = 0.5$ and a fixed policy.

- From $A$, with probability $0.5$ we stay in $A$ with reward $0$, and with probability $0.5$ we go to $B$ with reward $2$.
- From $B$ we always go to $A$ with reward $3$.

$$
v(A) = 0.5\,\big(0 + 0.5\,v(A)\big) + 0.5\,\big(2 + 0.5\,v(B)\big)
\qquad
v(B) = 3 + 0.5\,v(A)
$$

Put $v(B)$ into the first equation.

$$
0.75\,v(A) = 1 + 0.25\,\big(3 + 0.5\,v(A)\big) = 1.75 + 0.125\,v(A)
\quad\Rightarrow\quad
v(A) = \frac{1.75}{0.625} = 2.8
$$

$$
v(B) = 3 + 0.5 \cdot 2.8 = 4.4
$$

Check by putting the numbers back: $0.5\,(0 + 1.4) + 0.5\,(2 + 2.2) = 2.8$.

<!--
Let students do the substitution on paper first. Then go through it on the board. For large state spaces we do not solve this by hand. That is where policy evaluation comes in, later today.
-->

---
layout: top-title
color: light
align: lt
title: The best policy
---

:: title ::

# Optimal Policies and Optimal Values

:: content ::

Policy $\pi$ is at least as good as $\pi'$ if $v_\pi(s) \ge v_{\pi'}(s)$ for every state.

The optimal value functions

$$
v_*(s) = \max_\pi v_\pi(s)
\qquad
q_*(s,a) = \max_\pi q_\pi(s,a)
$$

Three facts to keep

- An optimal policy always exists for a finite MDP.
- There can be many optimal policies, but they all share the same $v_*$ and $q_*$.
- At least one optimal policy is deterministic.

<!--
The final arrows of the game are one such policy for the grid without dice.
-->

---
layout: top-title
color: light
align: lt
title: The best policy, as an equation
---

:: title ::

# The Bellman Optimality Equation

:: content ::

$$
v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma\, v_*(s') \right]
$$

$$
q_*(s,a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \max_{a'} q_*(s', a') \right]
$$

The max makes these equations nonlinear, so we cannot solve them like the two-state example.

Once we know $v_*$ and $p$, the best action in $s$ is

$$
\pi_*(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma\, v_*(s') \right]
$$

Once we know $q_*$, it is $\arg\max_a q_*(s,a)$ and we do not need $p$.

<AdmonitionType type='tip'>
At the end of the game no square wanted to switch, i.e. every card was already the max over the 4 moves. So the cards solve this equation and the final arrows are optimal.
</AdmonitionType>

<!--
The step from "the cards solve the equation" to "the cards are v*" needs the solution to be unique. It is unique when gamma < 1. In the grid gamma = 1, and it is still unique because any set of arrows that loops forever has a value of minus infinity.
-->

---
layout: top-title
color: light
align: lt
title: Where we are
---

:: title ::

# From Equations to Updates

:: content ::

We now turn the Bellman equations into update rules. This is **dynamic programming** (DP).

| | By hand | Today |
|---|---|---|
| What we have | Equations for $v_\pi$ and $v_*$ | Update rules that compute them |
| How we solve | Substitution, for 2 states | Sweeps, for many states |
| What we need | The model $p(s', r \mid s, a)$ | The same model |

Dynamic programming needs two things:

1. A **known model**, i.e. we know $p(s', r \mid s, a)$. Weeks 5 to 7 drop this need.
2. A state space small enough to **store one number per state**. Weeks 9 to 13 drop this need.

---
layout: top-title
color: light
align: lt
title: Step 1, how good is my policy
---

:: title ::

# Policy Evaluation

:: content ::

**Goal**: given a policy $\pi$, compute $v_\pi$.

Take the Bellman equation for $v_\pi$ and use it as an update rule. The old values are on the right and the new value is on the left.

$$
v_{k+1}(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma\, v_k(s') \right]
$$

- Start from any $v_0$, for example all zeros. Keep terminal states at 0.
- Each sweep updates every state once.
- As $k$ grows, $v_k$ gets closer to $v_\pi$.

<AdmonitionType type='important'>
This is a card round. With dice, the new card is an average over where the robot might land.
</AdmonitionType>

---
layout: top-title
color: light
align: lt
title: Back to the two states
---

:: title ::

# Policy Evaluation on the Two-State Example

:: content ::

By hand we got $v(A) = 2.8$ and $v(B) = 4.4$. Now we get them by sweeps. Both updates use the values of sweep $k$, as in the game.

$$
v_{k+1}(A) = 0.5\,\big(0 + 0.5\,v_k(A)\big) + 0.5\,\big(2 + 0.5\,v_k(B)\big)
\qquad
v_{k+1}(B) = 3 + 0.5\,v_k(A)
$$

| Sweep $k$ | 0 | 1 | 2 | 3 | 4 | 5 | 10 |
|---|---|---|---|---|---|---|---|
| $v_k(A)$ | 0 | 1.00 | 2.00 | 2.38 | 2.59 | 2.70 | 2.80 |
| $v_k(B)$ | 0 | 3.00 | 3.50 | 4.00 | 4.19 | 4.30 | 4.40 |

No algebra and no matrix inverse. We just repeat a simple update.

<!--
Ask students to do sweep 1 and sweep 2 on paper before showing the table.
For n states, solving the linear system directly costs about n^3. One sweep costs about n^2 times the number of actions. The number of sweeps depends on gamma and on the accuracy we want, not on n.
-->

---
layout: top-title
color: light
align: lt
title: The algorithm
---

:: title ::

# Iterative Policy Evaluation

:: content ::

```text
Input: policy π, small threshold θ > 0
Set V(s) = 0 for every state s, including the terminal states

repeat
    Δ = 0
    for each non-terminal state s:
        old = V(s)
        V(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [ r + γ V(s') ]
        Δ = max(Δ, |old − V(s)|)
until Δ < θ

return V   (close to v_π)
```

- This code works **in place**. It writes the new value over the old one right away, so later states in the same sweep already see it. This uses less memory and is often faster.
- With **two arrays**, we keep the old values in one array and write the new values in another. This is what we did in the game.

<!--
Both versions converge to v_pi. The order of the states matters for the in-place version. In the grid, sweeping from the star outward is faster than sweeping toward it.
-->

---
layout: top-title
color: light
align: lt
title: Why it works
---

:: title ::

# Why Do the Sweeps Converge?

:: content ::

One sweep shrinks the largest error over all states by at least a factor $\gamma$.

$$
\max_s \big| v_{k+1}(s) - v_\pi(s) \big| \;\le\; \gamma \, \max_s \big| v_k(s) - v_\pi(s) \big|
$$

- With $\gamma = 0.9$, each sweep keeps at most 90% of the error. After 50 sweeps less than 1% is left.
- With $\gamma = 1$ this bound says nothing. The sweeps still converge if every episode ends for sure under $\pi$, as with the arrows of our game.
- The starting values do not matter. The sweeps forget them.

<AdmonitionType type='tip'>
In the two-state example the discount is 0.5. The largest error over A and B goes 4.4, 1.8, 0.9, 0.43, 0.21, so each sweep at least halves it.
</AdmonitionType>

<!--
The formal name is contraction mapping. Sutton and Barto, Section 4.1, states when the sweeps converge, i.e. gamma < 1 or every episode ends under pi. David Silver's UCL lecture 3, Planning by Dynamic Programming, shows the contraction argument.
The error of a single state can shrink more slowly. v(A) goes from an error of 2.8 to 1.8 in the first sweep. The bound is about the largest error.
-->

---
layout: top-title
color: light
align: lt
title: Step 2, can I do better
---

:: title ::

# Policy Improvement

:: content ::

We know $v_\pi$. In state $s$, what if we take action $a$ once and then go back to $\pi$?

$$
q_\pi(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma\, v_\pi(s') \right]
$$

If some action has $q_\pi(s, a) > v_\pi(s)$, switching to it in $s$ gives a better policy.

So do it in every state at once. This is the **greedy policy**. Here $\pi'(s)$ is the one action that a deterministic policy picks in $s$.

$$
\pi'(s) = \arg\max_a \, q_\pi(s, a)
$$

**Policy improvement theorem**: the greedy policy is never worse. $v_{\pi'}(s) \ge v_\pi(s)$ for every state $s$.

<AdmonitionType type='important'>
If no action beats the current one in any state, the values already solve the Bellman optimality equation, so the policy is optimal.
</AdmonitionType>

---
layout: top-title
color: light
align: lt
title: Evaluate, improve, repeat
---

:: title ::

# Policy Iteration

:: content ::

$$
\pi_0 \xrightarrow{\;E\;} v_{\pi_0} \xrightarrow{\;I\;} \pi_1 \xrightarrow{\;E\;} v_{\pi_1} \xrightarrow{\;I\;} \pi_2 \xrightarrow{\;E\;} \cdots \xrightarrow{\;I\;} \pi_* \xrightarrow{\;E\;} v_*
$$

```text
Start with any policy π
repeat
    V  = policy_evaluation(π)              # E step: card rounds
    π' = greedy policy with respect to V   # I step: arrow round
    if π' == π: stop
    π = π'
return π, V
```

- Each new policy is better in at least one state or nothing changes and we stop.
- There are finitely many deterministic policies, so policy iteration stops after a finite number of steps, usually very few.
- Start each evaluation from the last $V$, not from zeros. In the game, one card round was enough after each switch.

<!--
A common bug: two actions with the same q value and the argmax picks a different one each time. The policy never stops changing. Break ties in a fixed order, or switch only when strictly better, as in the game.
-->

---
layout: top-title
color: light
align: lt
title: Policy iteration with dice
---

:: title ::

# Policy Iteration on Last Week's Grid

:: content ::

Slippery floor ($\tfrac{4}{6}$ the arrow, $\tfrac{1}{6}$ each side) and $\gamma = 1$. Start with **→ in every square** and evaluate it.

**$\pi_0$**: all →

| | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| **Row 1** | → -1.4 | → 0.8 | → 3.1 | → 4.0 |
| **Row 2** | 🟢 → -3.9 | 💥 → 2.3 | → 5.7 | ⭐ |
| **Row 3** | → -5.2 | → -4.0 | 💥 → 2.1 | → 4.0 |

---
layout: top-title
color: light
align: lt
title: Policy iteration with dice
---

:: title ::

# Improve, Then Evaluate Again

:: content ::

**$\pi_1$**: 6 arrows change and every value goes up.

| | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| **Row 1** | → 0.4 | → 2.3 | ↓ 4.9 | ↓ 7.8 |
| **Row 2** | 🟢 ↑ -1.2 | 💥 → 3.2 | → 6.3 | ⭐ |
| **Row 3** | ↑ -2.7 | → -2.5 | 💥 ↑ 3.9 | ↑ 6.6 |

---
layout: top-title
color: light
align: lt
title: Policy iteration with dice
---

:: title ::

# One More Step and We Stop

:: content ::

**$\pi_2$**: 2 arrows change. One more improvement step changes nothing so we stop.

| | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| **Row 1** | → 1.9 | → 3.9 | → 6.6 | ↓ 8.1 |
| **Row 2** | 🟢 ↑ 0.1 | 💥 → 3.8 | → 6.7 | ⭐ |
| **Row 3** | ↑ -1.5 | → -1.8 | 💥 → 4.5 | ↑ 6.7 |

This is exactly the table of best arrows from last week.

<!--
Two improvement steps changed arrows and a third one confirmed that nothing changes. With theta = 0.01 and in-place sweeps that start from the last V, the three evaluations took about 27, 12 and 10 sweeps.
Point out that the values only go up from one table to the next. That is the policy improvement theorem at work.
-->

---
layout: top-title
color: light
title: Apply it
---

:: title ::

# Paper: The AI Clinician for Sepsis (Nature Medicine 2018)

:: content ::

Komorowski et al. learned how to treat sepsis in intensive care from past hospital records.

| Piece | In the paper |
|---|---|
| Data | About 17,000 ICU patients from MIMIC-III, then tested on about 79,000 from eRI |
| State | 48 measurements per 4-hour step, grouped by k-means into 750 clusters |
| Action | IV fluid dose × vasopressor dose, 5 levels each, so 25 actions |
| Reward | $+100$ if the patient is alive at 90 days, $-100$ if not, only at the end |
| Model and solver | Transitions counted from the records, then policy iteration with $\gamma = 0.99$ |

Nobody gave them the model $p$. They estimated it from data, then ran policy iteration on it.

<!--
Two more states mark the end of a stay, one for death and one for discharge, so the MDP has 752 states.
The authors built 500 such models with different clusterings and kept the best one on validation data.
Evaluation is off-policy, with weighted importance sampling, because the learned policy cannot be tested on real patients.
-->

---
layout: top-title
color: light
title: Apply it
---

:: title ::

# Questions for the Sepsis Paper

:: content ::

1. Is a cluster of patients a Markov state? What could be missing?
2. Why is the reward given only at the end? What would go wrong with a reward for each normal lab value?
3. The model is counted from records. What happens to a (state, action) pair that doctors almost never chose?
4. The policy cannot be tried on real patients. How did the authors evaluate it, and how much would you trust that?
5. Which steps of policy iteration would change if there were 750,000 states instead of 750?
6. What would your project mentor, as a domain expert, need to check before trusting a policy like this?

<AdmonitionType type='tip'>
Read the follow-up critique by Jeter et al. (2019), "Does the Artificial Intelligence Clinician learn optimal treatment strategies for sepsis in intensive care?" It shows how a policy can look good on paper for the wrong reasons.
</AdmonitionType>

<!--
Point to Gottesman et al., "Guidelines for reinforcement learning in healthcare", Nature Medicine 2019, for students who want more.
For balance, the original authors answered the critique in "Understanding the Artificial Intelligence Clinician and optimal treatment strategies for sepsis in intensive care" (arXiv 1903.02345). They say the AI was limited to actions that clinicians took often enough.
Question 3 is the key one. Rare state-action pairs have a model built from very few records, and the greedy step may pick exactly those actions because their estimates are noisy.
-->

---
layout: top-title
color: light
align: lt
title: Avoid these
---

:: title ::

# Common Mistakes

:: content ::

- Mixing up a return and a value. One run is not an average.
- Forgetting the sum over $s'$. With dice, the next state is not fixed.
- Updating terminal states. Their value stays 0.
- Mixing old and new values by accident when you meant to use two arrays.
- Forgetting the $\gamma$ or the reward inside the bracket.
- Breaking ties in a different way each time so policy iteration never stops.
- Using $\gamma = 1$ to evaluate a policy that can loop forever. Its values never settle.

---
layout: top-title
color: light
align: lt
title: Your turn
---

:: title ::

# Practice Check

:: content ::

1. In the two-state example, what is $v(A)$ if $\gamma = 0$?
2. Start the two-state example from $v(A) = 2$ and $v(B) = 2$ and do one sweep of policy evaluation by hand.
3. In the game, why did (3,1) switch its arrow only after 🟢 switched?
4. Why is policy iteration sure to stop after a finite number of steps?
5. Why is $q_*$ more useful than $v_*$ when we do not know the dynamics?
6. In the sepsis paper, which part of the DP setup was not given and had to be estimated?

---
layout: grid-cards
cols: 4
---

:: card-1 ::
#### Bellman equation
A value written with the values of the next states

:: card-2 ::
#### Bellman optimality
The same, with a max over actions

:: card-3 ::
#### Optimal policy
The policy with the largest value in every state

:: card-4 ::
#### Dynamic programming
Compute values from a known model with the Bellman equations

:: card-5 ::
#### Policy evaluation
Repeat the Bellman update for $v_\pi$ until it settles

:: card-6 ::
#### Policy improvement
Act greedily with respect to $v_\pi$. It is never worse.

:: card-7 ::
#### Policy iteration
Evaluate, improve, repeat until the policy stops changing

:: card-8 ::
#### Model
$p(s', r \mid s, a)$, which DP needs and real problems rarely give

---
layout: top-title
color: sky-light
align: lt
title: Before next week
---

:: title ::

# Before We Meet Again

:: content ::

1. **Read Sutton and Barto, Sections 3.5 to 3.6 and 4.1 to 4.3**.
2. **Write a short response to the sepsis paper**, with a focus on the questions on the slide.
3. **Read the Go paper for week 5**: Silver et al., Nature 529, 484-489 (2016).
4. **Project teams**: the proposal and the mentor agreement are due this week. The MDP spec, the Gymnasium environment, and a non-RL baseline are due in week 7.

Next lecture: value iteration and then learning values from played episodes without a model.

<AdmonitionType type='tip'>
Ask yourself now if your project environment can give you the full model of its dynamics. If it can, DP is a strong baseline.
</AdmonitionType>

---
layout: section
title: Questions
class: text-center
---

# Learn More

[Course Homepage](https://github.com/m-fakhry/DSAI-402-RL)
