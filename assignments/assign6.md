# Assignment 6

## Reading 

- Read Section 6.1 from the main book
- Watch this video [Temporal Difference Explained – The Key to Q-Learning, by Super Data Science](https://www.youtube.com/watch?v=yaIYa9TP780&t=1s)
- ... and this video [Temporal Difference Learning (including Q-Learning) | Reinforcement Learning Part 4, by Mutual Information](https://www.youtube.com/watch?v=AJiG3ykOxmY&t=953s)

## Implementation 

**Replicating Temporal Difference Learning Visualization**

In the video “Temporal Difference Learning (including Q-Learning) | Reinforcement Learning Part 4” by Mutual Information, between minutes 3 and 6, there is a very nice animation of the episodes in the lower panel, accumulated rewards in the upper panel, and the evolving value function on the right lower panel.
Your task is to replicate a similar demonstration with the following requirements:
- Generate multiple episodes of state and reward sequences resembling those shown in the video (they don’t need to be exact replicas, but should be similar in spirit). This is similar to the previous assignment. 
- Implement one-step and 2-step Temporal Difference (TD) learning as presented in the lecture, showing how the value function is updated incrementally as the episode progresses
