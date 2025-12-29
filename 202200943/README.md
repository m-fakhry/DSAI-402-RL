# RL Tool - Reinforcement Learning Playground

Interactive web-based tool for experimenting with reinforcement learning algorithms.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Baher-Nader/RL-Tool.git
cd rl-tool
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv myenv
myenv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv myenv
source myenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Web Interface

```bash
cd web/backend
python app.py
```

The web interface will be available at: **http://localhost:5000**

### 5. Use the Interface

1. Open your browser and go to `http://localhost:5000`
2. Select an environment (GridWorld, FrozenLake, CartPole, MountainCar)
3. Select an algorithm (Policy Iteration, Value Iteration, Q-Learning, SARSA, Monte Carlo, N-Step TD)
4. Adjust parameters (gamma, alpha, epsilon, episodes)
5. Click "Train Agent" to start training
6. View visualizations and results

## Running Tests

Test individual algorithms:

```bash
python tests/test_policy_iteration.py
python tests/test_value_iteration.py
python tests/test_qlearning.py
python tests/test_sarsa.py
python tests/test_monte_carlo.py
python tests/test_nstep_td.py
```

## Project Structure

```
rl-tool/
├── algorithms/      # RL algorithm implementations
├── core/           # Base classes (Agent, Environment)
├── envs/           # Environment implementations
├── web/            # Web interface (Flask backend + frontend)
│   ├── backend/   # Flask API server
│   └── frontend/   # HTML/CSS/JavaScript UI
└── tests/         # Algorithm test scripts
```

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies


