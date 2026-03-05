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
lecture: 3
slide:
  disableSlideNumbers: true
slide_info: false
---

# Reinforcement Learning <br> (DSAI 402)
## Lecture 3

Mohamed Ghalwash
<Email v="mghalwash@zewailcity.edu.eg" />

---
transition: fade-out
layout: top-title
class: ns-c-center-item
---

:: title :: 

# Lecture 2 Recap

:: content :: 

- MDP 
- Bellman Equations 
- Optimal Policy 
  
  
---
layout: top-title
---

:: title :: 

# Example: Vending Machine

:: content :: 

Imagine an agent deciding which button (A, B, C) to press on a vending machine:
<!-- <div class="ns-c-tight"> -->

- **States** represent the machine’s current configuration (e.g., no selection made yet)
- **Actions**: pressing one of three buttons: A -> Biscoff, B -> Doritos, C -> Galaxy
- **Rewards**: based on the tastiness of each snack: Biscoff= 1, Doritos = 2, and Galaxy = 3
- The environment is deterministic: pressing a button always results in receiving the corresponding snack and episode ends
<!-- </div>  -->


<!-- style="display: grid; grid-auto-flow: column; place-items: center;" -->
<div class="grid place-items-center grid-cols-3" >
  
  <div v-click="1" class="col-span-1">

```mermaid 
flowchart TD
    Start([Start State])
    SnackA([Biscoff<br/>Reward: 1])
    SnackB([Doritos<br/>Reward: 2])
    SnackC([Galaxy<br/>Reward: 3])

    Start -->|Press A| SnackA
    Start -->|Press B| SnackB
    Start -->|Press C| SnackC

    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style SnackA fill:#bbf,stroke:#333,stroke-width:1px
    style SnackB fill:#bbf,stroke:#333,stroke-width:1px
    style SnackC fill:#bbf,stroke:#333,stroke-width:1px
```
  </div>

  <div v-click="2" class="col-span-2"> 

  <v-clicks>

  - [What is really missing from the formulation?]{.bg-red-200}

  - [Poilcy]{.bg-indigo-200} $\pi ( a | s )$

  - $\pi ( a = A | s ) = 0.2$, $\pi ( a = B | s ) = 0.3$, $\pi ( a = C | s ) = 0.5$

  </v-clicks>

  </div>

</div>


---
layout: top-title
---

:: title :: 

# Example: Vending Machine

:: content :: 

<div class="text-align-center" >

```mermaid {scale: 0.7}
flowchart TD
    Start([Start State])
    SnackA([Biscoff<br/>Reward: 1])
    SnackB([Doritos<br/>Reward: 2])
    SnackC([Galaxy<br/>Reward: 3])

    Start -->|Press A -- 0.2| SnackA
    Start -->|Press B -- 0.3| SnackB
    Start -->|Press C -- 0.5| SnackC

    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style SnackA fill:#bbf,stroke:#333,stroke-width:1px
    style SnackB fill:#bbf,stroke:#333,stroke-width:1px
    style SnackC fill:#bbf,stroke:#333,stroke-width:1px
```
</div> 

What is the value of the start state $v_\pi(s)$?

<v-clicks> 

- $\pi ( a = A | s ) = 0.2$, $\pi ( a = B | s ) = 0.3$, $\pi ( a = C | s ) = 0.5$
  - $v_\pi(s) = 0.2 * 1 + 0.3 * 2 + 0.5 * 3 =$ ==$2.3$==

- $\pi ( a = A | s ) = 1$, $\pi ( a = B | s ) = 0$, $\pi ( a = C | s ) = 0$
  - $v_\pi(s) = 1 * 1 + 0 * 2 + 0 * 3 =$  ==$1$==

- $\pi ( a = A | s ) = 0$, $\pi ( a = B | s ) = 0$, $\pi ( a = C | s ) = 1$
  - $v_\pi(s) = 0 * 1 + 0 * 2 + 1 * 3 =$ ==$3$==

</v-clicks>

<v-click>

<SpeechBubble position="l" color='rose-light' shape="round"  v-drag="[600,350,200,80]">
Which policy is the optimal one? 
</SpeechBubble>

</v-click> 


---
layout: top-title
---

:: title :: 

# Example: Not Greedy

:: content :: 

<div class="text-align-center" >

```mermaid {scale: 1.3}
flowchart TD
    Start([Start State])
    SnackA([S<br/>Reward: 1])
    SnackB([T<br/>Reward: 3])
    SnackC([H<br/>Reward: 2])
    SnackD([K<br/>Reward: -20])
    SnackE([E<br/>Reward: -15])

    Start --> SnackA
    Start --> SnackB
    Start --> SnackC
    SnackB --> SnackD
    SnackC --> SnackE

    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style SnackA fill:#bbf,stroke:#333,stroke-width:1px
    style SnackE fill:#bbf,stroke:#333,stroke-width:1px
    style SnackD fill:#bbf,stroke:#333,stroke-width:1px
```
</div> 




---
layout: section
--- 

# Dynamic Programming 


---
layout: top-title 
---

:: title :: 

# Policy Evaluation 

:: content :: 

- The existence and uniqueness of $v_\pi$ are guaranteed as long as either $\gamma < 1$

- Iterative policy evaluation: the sequence $\{v_k\}$  can be shown in general to converge to $v_\pi$  as $k\rightarrow \infty$

- Expected update: two arrays or one array (in place)

```python {1|2|3|4|5,6|7,8,9|7,8,9,10|11,12|all}
function iterative_policy_evaluation() {
    v = {s: 0 for s in env.states}
    while True:
        for s in env.states:
            old_v[s] = v[s]
            new_v[s] = 0
            for a in policy[s]:
                for s_ , r, prob in env.transitions(s, a):
                    new_v[s] += policy[s][a] * prob * (r + gamma * v[s_])
            v[s] = new_v[s]
        if max(abs(old_v - v[s])) < theta:
            break
    return v
}
```

<SpeechBubble style="font-family: 'Arial', sans-serif; font-size: 12px;" position="l" color='fuchsia-light' shape="round"  v-drag="[600,320,300,60]" class="custom-angle" animation="float" v-click="5" v-click.hide="6">

$v_\pi(s) = \sum_a \pi(a|s) \sum_{s^\prime} p(s^\prime|s,a) \left[ r(s,a,s^\prime) + \gamma v_\pi(s^\prime) \right]$

</SpeechBubble>



---
layout: top-title-two-cols
align: l-lm-lb
---

:: title :: 

# Example 

:: left :: 

![alt text](./images/3_iterative_policy_1.png){width=100%}

:: right :: 

<v-click>

![alt text](./images/3_iterative_policy_2.png){width=100%}

</v-click>

---
layout: center
class: text-center
---

# Learn More

[Course Homepage](https://github.com/m-fakhry/DSAI-402-RL)
