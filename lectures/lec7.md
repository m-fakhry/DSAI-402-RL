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
lecture: 7
slide:
  disableSlideNumbers: true
slide_info: false
---

# Reinforcement Learning <br> (DSAI 402)
## Lecture 7

Mohamed Ghalwash
<Email v="mghalwash@zewailcity.edu.eg" />

---
layout: fact
---

# Recording is NOT allowed 

---
transition: fade-out
layout: top-title
class: ns-c-center-item
---

:: title :: 

# Lecture 6 Recap

:: content :: 

- Monte-Carlo learns value functions or policies by sampling episodes under the current policy, starting from a fixed start state (or distribution)

- Monte-Carlo methods 
  - First-visit 
  - Every-visit

---
layout: cover
--- 

# Temporal Difference 

"If one had to identify one idea as central and novel to reinforcement learning, it would
undoubtedly be temporal-difference (TD) learning", Richard Sutton


---
layout: top-title
---

:: title ::

# Temporal Difference (TD) Learning 

:: content :: 

Before going deep into TD, most of the RL algorithms loop over two steps: 
- policy evaluation (aka prediction) 
- policy improvement (aka control)

The key distinction among these methods is their approach to **policy evaluation**

---
layout: top-title
---

:: title :: 

# What is Temporal Difference (TD)? 

:: content :: 

It is a mix between Monte Carlo and Dynamic Programming 

- It can learn directly from raw experience without a model of the environment’s dynamics

- It updates estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap)

```mermaid {theme: 'neutral', scale: 0.7}
graph LR
    RL[Reinforcement Learning]
    RL --> MB[Model-Based]
    RL --> MF[Model-Free]

    MB --> DP[Dynamic Programming]
    DP --> PI[Policy Iteration]
    DP --> VI[Value Iteration]

    MF --> MC[Monte Carlo]
    DP --> TD[Temporal Difference]
    MF --> TD
```

--- 
layout: top-title-two-cols
---

:: title ::

# MC Example 

:: left :: 

Grid world $4\times 4$
<v-click>

- Start state = $1$
- Terminal state = $16$
- Episode = \[$1, 2, 6, 10, 14, 15, 16$\]
- $\gamma=1$ 
</v-click>  

:: right :: 

<v-switch>

  <template #1> 
<div>
  <img src="./images/lec7_states_1.png" style="margin: auto; width: 90%;" />
</div>
  </template>

  <template #2> 
<div>
  <img src="./images/lec7_states_2.png" style="margin: auto; width: 90%;" />
</div>
  </template>

  <template #3> 
<div>
  <img src="./images/lec7_mc_1.png" style="margin: auto; width: 90%;" />
</div>
  </template>

  <template #4> 
<div>
  <img src="./images/lec7_mc_2.png" style="margin: auto; width: 90%;" />
</div>
  </template>

  <template #5> 
<div>
  <img src="./images/lec7_mc_3.png" style="margin: auto; width: 90%;" />
</div>
  </template>

</v-switch>


--- 
layout: top-title-two-cols
---

:: title ::

# Example 

:: left :: 

$$
V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]
$$


<v-click>

- States arranged in a grid
- Agent moves through states, receiving rewards
- Initial value estimates $V(s) = 0$
- Value updated step-by-step using TD rule
- $\alpha=0.5$

</v-click>  

:: right :: 

<v-switch>
  <template #1> 
<div>
  <img src="./images/lec7_td_values_step_1.png" style="margin: auto; width: 90%;" />
</div>
  </template>

  <template #2> 
<div>
  <img src="./images/lec7_td_values_step_2.png" style="margin: auto; width: 90%;" />
</div>
  </template>

  <template #3> 
<div>
  <img src="./images/lec7_td_values_step_3.png" style="margin: auto; width: 90%;" />
</div>
  </template>

  <template #4> 
<div>
  <img src="./images/lec7_td_values_step_4.png" style="margin: auto; width: 90%;" />
</div>
  </template>

  <template #5> 
<div>
  <img src="./images/lec7_td_values_step_5.png" style="margin: auto; width: 90%;" />
</div>
  </template>

  <template #6> 
<div>
  <img src="./images/lec7_td_values_step_6.png" style="margin: auto; width: 90%;" />
</div>
  </template>

</v-switch>

--- 
layout: top-title
---
:: title ::

# What is TD Learning?

:: content :: 

$$
v(s_t) = v(s_t) + \alpha \left[ \textcolor{red}{R_{t+1} + \gamma v(s_{t+1})} - v(s_t) \right]
$$


$\alpha$ is the learning rate


<v-click>

- TD Error: $\delta_t = R_{t+1} + \gamma v(s_{t+1}) - v(s_t)$
- Measures the difference between actual and estimated returns
- Used to adjust $v(s_t)$ closer to true expected future rewards
- Combines ideas from **Monte Carlo** and **Dynamic Programming**
- Updates estimates **at each step**, not waiting for episode end
<!-- - Learns the **value function** $v(s)$ of states directly -->
- Model-free reinforcement learning method

</v-click>

--- 
layout: top-title
---

:: title ::

# Comparison 

:: content :: 

$${0|1|1,2,3|1,2,3,4,5|all}
\begin{array}{rlll}
\text{TD} & \Rightarrow \qquad & 
v(s_t) & = v(s_t) + \alpha \left[ \textcolor{red}{R_{t+1} + \gamma v(s_{t+1})} - v(s_t) \right]\\
&&&&\\
\text{MC} & \Rightarrow \qquad & 
v_(s_t) & = \mathbb{E} \left[ \textcolor{red}{R_{t+1} + \gamma G_{t+1}} \mid S_t = s \right] 
\\
&&&&\\
\text{Value Iteration} & \Rightarrow \qquad & 
v_\pi(s)  & = \max_a \sum_{s^\prime} p(s^\prime|s,a) \textcolor{red}{\left[ r(s,a,s^\prime) + \gamma v_\pi(s^\prime) \right]} 
\\
&&&&\\
\text{Policy Evaluation} & \Rightarrow \qquad & 
v_\pi(s)  & = \sum_a \pi(a|s) \sum_{s^\prime} p(s^\prime|s,a) \textcolor{red}{\left[ r(s,a,s^\prime) + \gamma v_\pi(s^\prime) \right]} 
\\
\end{array}
$$

--- 
layout: top-title
---

:: title ::

# Why TD Learning?

:: content :: 

- Can learn **online**: updates immediately at every step
- Does not require knowledge of the full episode
- Often faster and more data-efficient than Monte Carlo methods

---
layout: image
image: ./images/TD_1.png
---


---
layout: top-title
--- 

:: title :: 

# $n$-step TD 

:: content :: 

One kind of intermediate method that would perform an update based on an intermediate number of rewards: more than one, but less than all of them until termination

$$
\begin{array}{ll}
v(s_t) & = v(s_t) + \alpha \left[ \textcolor{red}{R_{t+1} + \gamma v(s_{t+1})} - v(s_t) \right]\\
& = v(s_t) + \alpha \left[ \textcolor{red}{G_{t:t+1}} - v(s_t) \right]
\end{array}
$$


$${0|1|1,2,3|all}
\begin{array}{rlll}
\text{1-step TD} & \Rightarrow \qquad & 
G_{t:t+1} & = R_{t+1} + \gamma v(s_{t+1})\\\\
\text{2-step TD} & \Rightarrow \qquad & 
G_{t:t+2} & = R_{t+1} + \gamma R_{t+2} + \gamma^2 v(s_{t+2})\\\\
\\
&&&&\\
\text{$n$-step TD} & \Rightarrow \qquad & 
G_{t:t+n} & = R_{t+1} + \gamma R_{t+2} + \ldots + \gamma^n v(s_{t+n})\\\\
\end{array}
$$
<v-click>

Then the update rule becomes 
$v(s_t) = v(s_t) + \alpha \left[ \textcolor{red}{G_{t:t+n}} - v(s_t) \right]$ 
</v-click>

---
layout: image
image: ./images/TD_2.png
---

---
layout: top-title
---

:: title :: 

# Algorithm 

:: content :: 

![algorithm](./images/TD_3.png){width=70%;margin=auto}

---
layout: top-title-two-cols
columns: is-4
--- 

:: title :: 

# Evaluate the Performance of a Model-Free Algorithm

:: right :: 

<v-switch>

  <template #1> 
<div>
  <img src="./images/TD_6.png" style="margin: auto; width: 90%;" />
</div>
  </template>

  <template #2> 
<div>
  <img src="./images/TD_7.png" style="margin: auto; width: 90%;" />
</div>
  </template>

  <template #3> 
<div>
  <img src="./images/TD_4.png" style="margin: auto; width: 90%;" />
</div>
  </template>


</v-switch>

:: left :: 

**Random Walk Example**
<!-- - Compute the error between the estimated value and the true value -->
- For each episode
  - update the value and compute the error (loss) 
<!-- - Plot error (loss) vs episode (epochs)  -->
- Average across multiple runs 
- Run for different $\alpha$ and $n$ 
<!-- - Check which one converge faster?  -->
<!-- - Study the effect of $\alpha$ 
- Study the effect of $n$  -->

---
layout: center
class: text-center
---

# Learn More

[Course Homepage](https://github.com/m-fakhry/DSAI-402-RL)
