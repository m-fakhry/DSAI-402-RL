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
lecture: 1
slide:
  disableSlideNumbers: true
slide_info: false
---

# Reinforcement Learning <br> (Spring DSAI 402)
## Lecture 1

Mohamed Ghalwash
<Email v="mghalwash@zewailcity.edu.eg" />

---
layout: fact
---

# Recording is NOT allowed 


---
transition: fade-out
layout: image-right
image: images/1_rl_bg.png
---

# What is Reinforcement Learning?

- Learning optimal decisions over time
- Machine learning vs RL 
- Supervised vs Unsupervised learning 


---
layout: image-right
image: images/1_mouse_maze.png
---

**Robot mouse maze world**

<v-clicks>

- The environment is a maze containing food and electricity at certain locations 
- The robot mouse can perform actions as turning left or right and moving forward  
- The robot can observe the entire state of the maze to decide which action to take 
- Its goal is to collect as much food as possible while avoiding electric shocks whenever it can
- The food and electricity signals act as rewards provided by the environment
- _Suffer from a slight electric shock to reach an area with abundant food_ **OR** _standing and gaining nothing_

</v-clicks>


---
layout: fact
---

## The agent’s ultimate goal is to maximize its total accumulated reward


---
layout: top-title
---
:: title :: 

# RL Objectives 

:: content :: 

- Learn methods that will allow our robot to learn on its own how to avoid electricity and gather as much food as possible
  - Reward is +ve when gather food, -ve from electric shocks, neutral otherwise. 
  - Relate reward to action, let the agent learns how to perform an action better

**Challenges in RL**

- No i.i.d assumption 
- Exploit and Explore 
- Delay between reward and action 


---
layout: side-title
titlewidth: is-3
color: white
align: rm-lm
---

:: title :: 

```mermaid {theme: 'neutral', scale: 1}
graph LR
A((Agent)) --> |Action| B((Environment))
B --> |Observation| A
B --> |Reward| A
classDef cir fill:#f9f,stroke:#333,stroke-width:4px;
class A,B cir;
linkStyle 0 stroke:#333,stroke-width:2px,color:black;
linkStyle 1 stroke:#333,stroke-width:2px,color:green;
linkStyle 2 stroke:#333,stroke-width:2px,color:blue;
```

:: content :: 

- Reward
- Agent 
- Observation 
- Environment 
- Actions 

<!-- :: note :: 

the term reinforcement comes from the fact that a reward obtained by an agent should reinforce its behavior in a positive or negative way -->

---
layout: top-title 
---

:: title :: 

# Markov (Reward) Process 

:: content :: 

- Markov Process 
  - states 
  - transition matrix 
- Markov Reward Process
  - states
  - transition matrix 
  - reward $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$
  - value $V(s) = E(G | S_t = s)$

---
layout: top-title
# image: images/mouse_maze.png
---

:: title :: 

# Markov Process 

:: content :: 

- States 
- Transition matrix $P$
- Markov Property: $P(S_t|S_{t-1},S_{t-2},\ldots,S_1) = P(S_t|S_{t-1})$

<br><br>
Examples:

- Weather `S R S S S R R R S R `

- Currency Exchange Rate 
  
<br><br>

What does it mean $P^2$? 
  

---
layout: top-title
# image: images/mouse_maze.png
---

:: title :: 

# Markov Reward Process 

:: content :: 

- States: $S$
- Transition matrix: $P$ _source $\rightarrow$ target_
- Accumulated Reward: $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$
- Markov property: $P(R_t, S_t|S_{t-1},S_{t-2},\ldots,S_1) = P(R_t, S_t|S_{t-1})$
- Value function: $V(s) = E(G | S_t = s)$

<br><br><br>
Examples 
- Cat moving through rooms 


---
layout: center
class: text-center
---

# Learn More

[Slidev](https://sli.dev) · [Course Homepage](https://github.com/m-fakhry/DSAI-402-RL)
