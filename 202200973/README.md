# RL Agent Trainer

An interactive Reinforcement Learning visualization studio built with Streamlit, Gymnasium, and Pygame.

## Features

- **Environments**:
  - Custom Breakout
  - FrozenLake-v1
  - CartPole-v1
  - MountainCar-v0 (with reward shaping)
- **Algorithms**:
  - **Tabular / Temporal Difference**: Q-Learning, SARSA, TD(0), n-step TD
  - **Monte Carlo**: First-visit MC
  - **Dynamic Programming** (FrozenLake only): Policy Evaluation, Policy Iteration, Value Iteration
- **Interactive Training**: Watch the agent learn in real-time or speed up training.
- **Visualization**: Live rendering of environment states and reward plotting.

## Installation

1.  **Clone or download** this repository.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

1.  **Select Environment**: Choose the game/task you want to solve.
2.  **Select Algorithm**: Choose the RL method.
3.  **Adjust Hyperparameters**: Tune alpha, gamma, epsilon, etc.
4.  **Train**: Click "Start New Training".
5.  **Watch**: Observe the training progress and final result.
