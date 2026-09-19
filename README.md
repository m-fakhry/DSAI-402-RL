# Reinforcement Learning (DSAI 402) - 2026-2027

Repository for the RL undergraduate course (DSAI 402) for the 2026-2027 academic year at Zewail City University.

This year the course 
- transitions the focus toward deep reinforcement learning, 
- anchors every algorithm to a real published application, and
- adds a class project built with a mentor from outside the department.

---

### Previous Offerings

- [Fall 2025-2026](https://github.com/m-fakhry/DSAI-402-RL/tree/fall2526)
- [Spring 2025-2026](https://github.com/m-fakhry/DSAI-402-RL/tree/spring2526)

---

### Logistics

Course | Reinforcement Learning - DSAI 402
---|----
Webpage| [https://github.com/m-fakhry/DSAI-402-RL](https://github.com/m-fakhry/DSAI-402-RL)
Instructor | Prof. Mohamed Ghalwash (mghalwash@zewailcity.edu.eg)
Structure | 2-hour lecture (Monday 2pm-4pm) and 2-hour lab (Tue 4-6, Wed 2-4, Wed 4-6)
TAs | Eng. Aya Nageh, Eng. Huda Ayman
Communication | Moodle / Email. No phone calls. No whatsapp. 
Lab Policy| Assignments, quizzes, and project milestones
Book | "_Reinforcement Learning, an Introduction_", Sutton and Barto, 2nd Edition, 2018
Supplementary Book|"_Deep Reinforcement Learning Hands-On_", Lapan, 2018
Objective | Understand the foundations of RL, which is a computational approach where an agent learns to maximize cumulative reward through interaction with an environment, and apply it to real problems across scientific and industrial domains
Prerequisites | Deep Learning, PyTorch, Probabilities
Tools/APIs | [Gymnasium](https://gymnasium.farama.org/), PyTorch, Weights & Biases OR MLFlow (optional)

---

### Course Learning Outcomes

CLO \# | Outcome | Statement
---|---|---
 1 | Foundations of RL | Explain the fundamental principles of reinforcement learning and identify real-world problems for which an RL formulation is appropriate
 2 | MDP Formulation | Formulate states, actions, transitions, and reward functions for complex decision-making scenarios drawn from real application domains
 3 | Exact Solution Methods | Apply Bellman equations and dynamic programming to compute optimal policies for problems with tractable state spaces
 4 | Learning from Experience | Design and implement tabular model-free agents (Monte Carlo, TD, Q-learning) for environments with unknown dynamics
 5 | Deep RL | Implement and train deep RL agents using function approximation, value-based methods, and policy gradients for high-dimensional problems
 6 | Evaluation & Appraisal | Evaluate and compare RL algorithms empirically against baselines, and critically appraise published RL applications across domains

---

### Lectures

Please note that the syllabus content is subject to change throughout the semester. Topics may be added or removed based on the instructor's discretion, student progress, and available time. Your feedback and participation will inform these adjustments to ensure alignment with course goals and schedule constraints.

Week| Date |Topic | Contents | Application Paper | CLO | Lecture | Assignment
---|---|---|---|---|---|---|---
1 | 09-21 | Why RL | Deployed RL systems across domains, sequential vs supervised learning, course project tracks | [Survey of deployed systems](https://github.com/VincentLiu3/real-world-RL-deployment) | 1 | [L1: Why RL](lectures/lecture1.md)  | [Assignment 1](assignments/assignment1.md)
2 | 09-28 | RL Basics | Agent, environment, action, reward, return, discounting, Markov process | [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/pdf/1003.0146) (2012) — _business_ | 1, 2, 6 |  | 
3 | 10-05 | MDP | Markov decision process, policy, value functions, Bellman equations | [A Graph Placement Methodology for Fast Chip Design](https://www.nature.com/articles/s41586-021-03544-w) (2021) — _nanotechnology, EE_ — _read the formulation only, not the algorithm_ | 2, 3, 6 |  | 
4 | 10-12 | Dynamic Programming | Policy evaluation, policy improvement, policy iteration, value iteration | [The Artificial Intelligence Clinician Learns Optimal Treatment Strategies for Sepsis in Intensive Care](https://www.nature.com/articles/s41591-018-0213-5) (2018) — _healthcare_ | 3, 6 |  | **Project proposal + mentor agreement**
5 | 10-19 | Monte Carlo | MC prediction, first/every visit, $q$ estimation, $\epsilon$-greedy control | [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://www.nature.com/articles/nature16961) (2016) — _games_ | 4, 6 |  |
6 | 10-26 | Temporal Difference | TD(0), $n$-step returns, MC vs TD | [Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) (1995) — _games_ | 4, 6 |  | 
7 | 11-02 | Q-Learning | Off-policy control, SARSA as on-policy contrast, exploration | [Improving Elevator Performance Using Reinforcement Learning](https://proceedings.neurips.cc/paper/1995/hash/390e982518a50e280d8e2b535462ec1f-Abstract.html) (1996) — _engineering_ | 4, 6 |  | **Environment spec + non-RL baseline**; lab quiz
8 | 11-09 | **Midterm** | | | 1, 2, 3, 4 | |
9 | 11-16 | Function Approximation | Why tables fail, linear FA, semi-gradient methods, the deadly triad | — | 4, 5 |  | 
10 | 11-23 | Deep Q-Networks | Neural Q-function, target network, experience replay, Atari | [Human-Level Control Through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236) (2015) — _games_; [Autonomous Navigation of Stratospheric Balloons Using Reinforcement Learning](https://www.nature.com/articles/s41586-020-2939-8) (2020) — _aerospace_ | 5, 6 |  | **Working RL agent**
11 | 11-30 | Policy Gradients | Policy parameterization, REINFORCE, baselines | [Molecular De-Novo Design Through Deep Reinforcement Learning](https://doi.org/10.1186/s13321-017-0235-x) (2017) — _bioinformatics_ | 5, 6 |  | lab quiz
12 | 12-07 | Actor-Critic | Advantage estimation, A2C, shared networks, continuous actions | [Magnetic Control of Tokamak Plasmas Through Deep Reinforcement Learning](https://www.nature.com/articles/s41586-021-04301-9) (2022) — _physics_ | 5, 6 |  |
13 | 12-14 | PPO and Alignment | Trust regions, PPO, RLHF, reward models, DPO | [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) (2022); [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (2023) — _LLMs_ | 5, 6 |  |
14 | 12-21 | **Project Demos** | Team presentations and evaluation | — | 2, 5, 6 | | **Final report due**; demos
15 | 12-28 | **Project Demos** | Team presentations and evaluation | — | 2, 5, 6 | | Demos
16 | | **Final** | | | all | |


---

### Class Project

- Teams of 3-4 apply RL to a problem from a domain outside data science. 
<!-- Full specification in [project/README.md](project/README.md). -->

- Each team recruits one mentor from another department (engineering, aerospace, bioinformatics, business, physics, nanotechnology, etc.) or from industry. The mentor supplies or validates the problem, sanity-checks the reward function, and gives feedback at two checkpoints. Mentors are not graded and are not required to know RL.

- Every team picks one of three tracks. All three share the same technical core: MDP formulation, Gymnasium-compatible environment, trained agent, comparison against a non-RL baseline, and honest evaluation (success and failure analysis).

    Track | Final artifact | Additional emphasis
    ---|---|---
    **Scientist** | A manuscript paper + repo | A real research question, ablations, seeds and variance, related work
    **Engineer** | Deployable system + repo | Robustness, reproducibility, tests, inference cost, working demo
    **Innovator / Entrepreneur** | Prototype + pitch + feasibility memo | Who has this problem, what they do today, why RL, what scaling requires


- Milestones

    Week | Deliverable
    ---|---
    4 | Proposal: problem, domain, track, mentor agreement
    7 | MDP specification, Gymnasium environment, non-RL baseline results
    10 | Working RL agent with training curves, mentor verification 
    14 | Final report and mentor comment; demos in weeks 14-15

---

### Grading Policy

Topic| Percentage | Notes
---|---|---
Lab Assignments | 15% | 8 assignments
Paper Responses | 5% | For the assigned papers
Lab Quizzes | 10% | Weeks 7 and 11
Class Project | 25% | Distributed across the four milestones
Midterm | 15% |
Final | 30% |

---

### Course Instructions

Principle: deadlines are firm.

**Submissions**

- All assignments must be uploaded to the Moodle system before the deadline, even if the assignment has already been graded in the lab.
- Assignment grades depend on discussing your work with your TA. There are no extensions on these discussions: if you miss the discussion for an assignment, you lose its grade.

**Excuses and make-up tasks**

- Medical excuses must be submitted within one week of the excused task (assignment, quiz, or midterm). Excuses submitted after that window will not be considered.
- Make-up tasks cover the material taught up to the date of the make-up, not the material of the original task.

**Grade petitions**

- Once coursework grades (assignment, quiz, etc.) are posted, you have one week to raise an issue. If you believe there is an error in your grade, email me and CC your TA with:

  1. A clear and detailed explanation of exactly why you are petitioning the grade.
  2. Any relevant and approved documentation supporting your request, if the petition concerns a missed assignment, quiz, or exam.

- No grade adjustments will be considered after this deadline, and petitions that do not follow the format above will not be reviewed.

---

### Policies

**Use of AI tools.** You may use AI assistants for debugging, visualization, and boilerplate. You may not use them to produce your MDP formulations, your algorithm implementations, or your written analysis. The learning is in the implementation — a working agent you did not build teaches you nothing. Disclose any AI assistance in your submission.

**Academic integrity.** Standard Zewail City policy applies. Project code must be your team's own; third-party environments and libraries are permitted and must be cited.

---

### Resources

- [Gymnasium documentation](https://gymnasium.farama.org/)
- [Sutton & Barto, 2nd Edition (free PDF)](http://incompleteideas.net/book/the-book-2nd.html)
- [MLFlow](https://mlflow.org/)