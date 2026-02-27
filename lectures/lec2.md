---
theme: neversink
background: RL-bg.png
class: 'text-center'
transition: slide-left
title: RL (DSAI 402)
author: Mohamed Ghalwash
year: Spring 2025-2026
venue: Zewail City
mdc: true
lecture: 2
slide:
  disableSlideNumbers: true
slide_info: false
---

# Reinforcement Learning <br> (DSAI 402)
## Lecture 2

Mohamed Ghalwash
<Email v="mghalwash@zewailcity.edu.eg" />

---
transition: fade-out
layout: top-title
class: ns-c-center-item
---

:: title :: 

# Lecture 1 Recap

:: content :: 

- Markov Process
  - Multiple chains 
  - Rolling average 
  
- Markov Reward Process
  - Accumulated (==discounted==) reward, $G_t = R_{t+1} + \gamma G_{t+1}$
  - Value function $v(s)$ = expected accumulated reward
    - How good it is for the agent to be in a given state
    - Monte Carlo: involve averaging over many random samples of actual returns
  

---
layout: top-title
---

:: title :: 

# Markov Decision Process

:: content :: 

- Markov Reward Process + Actions
- Transition matrix: $P$ (_source and ==action==_) $\rightarrow$ (_target_)
- Markov: probabilistic move $P(R_t, S_t|S_{t-1},A_{t-1})$, i.e. $p(s^\prime | s,a)$, $r(s,a,s^\prime)$
  
<div style="text-align:center;">
  <img src="./images/2_actions.png" alt="probabilistic actions" style="display: block; margin-left: auto; margin-right: auto; width: 80%;" />
</div>

---
layout: top-title
---

:: title :: 

# What is Policy?

:: content :: 

- $\pi(a|s)$
  - a mapping from states to probabilities of selecting each possible action

<div style="text-align:center;">

```mermaid {theme: 'neutral', scale: 0.9}
%% States: s1, s2; Actions: a1, a2
%% Each cell shows \pi(a|s)
flowchart TB
  subgraph Policy_Matrix["$$\pi(a|s)$$"]
    direction LR
    S1A1["$$\pi(a_1|s_1)$$"]
    S1A2["$$\pi(a_2|s_1)$$"]
    S2A1["$$\pi(a_1|s_2)$$"]
    S2A2["$$\pi(a_2|s_2)$$"]
  end
  S1(("$$s_1$$"))
  S2(("$$s_2$$"))
  A1(("$$a_1$$"))
  A2(("$$a_2$$"))
  S1 -- " " --> S1A1
  S1 -- " " --> S1A2
  S2 -- " " --> S2A1
  S2 -- " " --> S2A2
  S1A1 -- " " --> A1
  S1A2 -- " " --> A2
  S2A1 -- " " --> A1
  S2A2 -- " " --> A2

  linkStyle 0,4 stroke:#1f77b4,stroke-width:3px;
  linkStyle 2,6 stroke:#d62728,stroke-width:3px;
  classDef redNode fill:#2e2eff,color:#fff,stroke:#000,stroke-width:2px;
  classDef blueNode fill:#E31243,stroke:#000,stroke-width:1px;
  class S1A1 redNode;
  class S2A1 blueNode;
```

</div>


---
layout: top-title
---

:: title :: 

# What is Policy?

:: content :: 

- $v_{\pi}(s)$
  - the state-value function of a state $s$ under a policy $\pi$
  - the expected return when starting in $s$ and following $\pi$ thereafter

<div style="text-align:center;">

```mermaid {theme: 'neutral', scale: 0.8}
%% State-value vector v_pi(s)
flowchart TB
  V1["$$v_\pi(s_1)$$"]
  V2["$$v_\pi(s_2)$$"]
  V3["$$v_\pi(s_3)$$"]
  V1 --> V2 --> V3
```

</div>


---
layout: top-title
---

:: title :: 

# What is Policy?

:: content :: 

- $q_{\pi}(s, a)$
  - the action-value function of taking action $a$ in state $s$ under a policy $\pi$ 
  - the expected return when performing action $a$ in state $s$ and following $\pi$ thereafter

<div style="text-align:center;">

```mermaid {theme: 'neutral', scale: 0.8}
%% Q(s,a) matrix for 2 states, 2 actions
flowchart TB
  subgraph Q_Matrix["Q_π(s,a)"]
    direction LR
    Q11["$$q_\pi(s_1,a_1)$$"]
    Q12["$$q_\pi(s_1,a_2)$$"]
    Q21["$$q_\pi(s_2,a_1)$$"]
    Q22["$$q_\pi(s_2,a_2)$$"]
  end
  S1(("$$s_1$$"))
  S2(("$$s_2$$"))
  A1(("$$a_1$$"))
  A2(("$$a_2$$"))
  S1 -- " " --> Q11
  S1 -- " " --> Q12
  S2 -- " " --> Q21
  S2 -- " " --> Q22
  Q11 -- " " --> A1
  Q12 -- " " --> A2
  Q21 -- " " --> A1
  Q22 -- " " --> A2

  linkStyle 0,4 stroke:#1f77b4,stroke-width:3px;
  linkStyle 2,6 stroke:#d62728,stroke-width:3px;
  classDef redNode fill:#2e2eff,color:#fff,stroke:#000,stroke-width:2px;
  classDef blueNode fill:#E31243,stroke:#000,stroke-width:1px;
  class Q11 redNode;
  class Q21 blueNode;
```

</div>


---
layout: top-title
class: text-center
---

:: title ::

# Bellman

:: content :: 

<div class="grid" style="display: grid; grid-template-columns: 3fr 1fr; gap: 1em;">
  <div>

$$
{1|1,2|2,3|3,4|4,5|all}
\begin{array}{ll}
v_\pi(s) & = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right] \\ \\
        & =  \mathbb{E}_\pi \left[ \textcolor{green}{R_{t+1} + \gamma G_{t+1}} \mid s \right] \\ \\
        & =  \mathbb{E}_\pi \left[ \textcolor{green}{R_{t+1} + \gamma v_\pi(s^\prime)} \mid s \right] \\ \\
        & = \sum_a \sum_{s^\prime} \textcolor{blue}{\pi(a|s)} \textcolor{red}{p(s^\prime|s,a)} \left[ \textcolor{green}{r(s,a,s^\prime) + \gamma v_\pi(s^\prime)} \right] \\ \\ 
        & = \sum_a \textcolor{blue}{\pi(a|s)} \sum_{s^\prime} \textcolor{red}{p(s^\prime|s,a)} \left[ \textcolor{green}{r(s,a,s^\prime) + \gamma v_\pi(s^\prime)} \right]
\end{array}
$$

  </div>
  <div>

```mermaid {theme: 'neutral', scale: 0.8}
graph TD
  S1(("$$s_1$$"))
  S3(("$$s_1^\prime$$"))
  S4(("$$s_1^{\prime\prime}$$"))
  A1(("$$a_1$$"))
  A2(("$$a_2$$"))
  S1 -- "$$\pi(a_1|s_1)$$" --> A1
  S1 -- "$$\pi(a_2|s_1)$$" --> A2
  A1 -- "$$p(s_1^\prime|s_1, a_1), r^\prime$$" --> S3
  A1 -- "$$p(s_1^{\prime\prime}|s_1, a_1), r^{\prime\prime}$$" --> S4
  
  %% Style links from S1 (blue)
  linkStyle 0 stroke:#1f77b4,stroke-width:3px;
  linkStyle 1 stroke:#1f77b4,stroke-width:3px;
  %% Style links from S2 (red)
  linkStyle 2 stroke:#d62728,stroke-width:3px;
  linkStyle 3 stroke:#d62728,stroke-width:3px;
  classDef blackNode fill:#000,color:#fff,stroke:#000,stroke-width:2px;
  classDef stateNode fill:none,stroke:#000,stroke-width:1px;
  class A1,A2 blackNode;
  class S1,S3,S4 stateNode;
```

  </div>
</div>


<v-click>

#### For each ==state-action-next state== triple, we calculate the joint probability $\textcolor{blue}{\pi(a|s)}\textcolor{red}{p(s^\prime|s, a)}$ and use it to weight the expected reward. Summing over all such triples yields the total expected state-value

</v-click>


---
layout: top-title-two-cols
columns: is-8
align: l-lt-lm
---

:: title :: 

# Value Functions (Bellman)

:: left :: 


$$
{all}
\begin{array}{ll}
v_\pi(s) & = \sum_a \textcolor{blue}{\pi(a|s)} \sum_{s^\prime} \textcolor{red}{p(s^\prime|s,a)} \left[ \textcolor{green}{r(s,a,s^\prime) + \gamma v_\pi(s^\prime)} \right]
\end{array}
$$

<br/> 

$$
{0|1|2|all}
\begin{array}{ll}
q_\pi(s, a) 
         & =  \mathbb{E}_\pi \left[ R_{t+1} + \gamma v_\pi(s^\prime) \mid s, a \right] \\ \\
         & = \sum_{s^\prime} \textcolor{red}{p(s^\prime|s,a)} \left[ r(s,a,s^\prime) + \gamma \textcolor{green}{v_\pi(s^\prime)} \right]
\end{array}
$$

<br/><br/>

$$
{0|all}
\begin{array}{ll}
v_\pi(s) 
        & = \sum_a \textcolor{blue}{\pi(a|s)} \sum_{s^\prime} \textcolor{red}{p(s^\prime|s,a)} \left[ \textcolor{green}{r(s,a,s^\prime) + \gamma v_\pi(s^\prime)} \right] \\ \\ 
        & = \sum_a \textcolor{blue}{\pi(a|s)} q_\pi(s, a) 
\end{array}
$$

:: right :: 


```mermaid {theme: 'neutral', scale: 0.8}
graph TD
  S1(("$$s_1$$"))
  S3(("$$s_1^\prime$$"))
  S4(("$$s_1^{\prime\prime}$$"))
  A1(("$$a_1$$"))
  A2(("$$a_2$$"))
  S1 -- "$$\pi(a_1|s_1)$$" --> A1
  S1 -- "$$\pi(a_2|s_1)$$" --> A2
  A1 -- "$$p(s_1^\prime|s_1, a_1), r^\prime$$" --> S3
  A1 -- "$$p(s_1^{\prime\prime}|s_1, a_1), r^{\prime\prime}$$" --> S4
  
  %% Style links from S1 (blue)
  linkStyle 0 stroke:#1f77b4,stroke-width:3px;
  linkStyle 1 stroke:#1f77b4,stroke-width:3px;
  %% Style links from S2 (red)
  linkStyle 2 stroke:#d62728,stroke-width:3px;
  linkStyle 3 stroke:#d62728,stroke-width:3px;
  classDef blackNode fill:#000,color:#fff,stroke:#000,stroke-width:2px;
  classDef stateNode fill:none,stroke:#000,stroke-width:1px;
  class A1,A2 blackNode;
  class S1,S3,S4 stateNode;
```


---
layout: top-title
---

:: title :: 

# Optimal Policy

:: content :: 


A policy $\pi^\prime$ is defined to be better than or equal to a policy $\pi$ if its expected return is greater than or equal to that of $\pi$ for all states.

<div style="text-align:center;">

$\pi^\prime >= \pi$ if and only if $v_{\pi^\prime}(s) >= v_{\pi}(s)$ for all $s \in S$

</div>

<br/><br/><br/>

There is always at least one policy that is better than or equal to all other policies. This is an optimal policy

<div style="text-align:center;">

$v_{*}(s) := max_\pi \; v_\pi(s)$

$q_{*}(s, a) := max_\pi \; q_\pi(s, a)$

</div> 

---
layout: statement
---

## A policy is a stochastic rule by which the agent selects actions as a function of states. The agent’s objective is to maximize the amount of reward it receives over time.



---
layout: center
class: text-center
---

# Learn More

[Course Homepage](https://github.com/m-fakhry/DSAI-402-RL)

<!-- https://gibberblot.github.io/rl-notes/single-agent/temporal-difference-learning.html -->