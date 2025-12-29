import streamlit as st
import time
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import gymnasium as gym
import pygame


# 1. CLASSES: RL AGENTS


class BaseAgent:
    def __init__(self, action_space_n, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(action_space_n))
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.action_space_n = action_space_n

    def choose_action(self, state, greedy=False):
        if greedy or (random.random() > self.epsilon):
            return np.argmax(self.q_table[state])
        return random.randint(0, self.action_space_n - 1)

    def update(self, *args):
        pass

class QLearningAgent(BaseAgent):
    def update(self, state, action, reward, next_state, done):
        old_val = self.q_table[state][action]
        next_max = 0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * next_max
        self.q_table[state][action] = old_val + self.alpha * (target - old_val)

class SARSAAgent(BaseAgent):
    def update(self, state, action, reward, next_state, done, next_action=None):
        old_val = self.q_table[state][action]
        next_val = 0 if done else self.q_table[next_state][next_action]
        target = reward + self.gamma * next_val
        self.q_table[state][action] = old_val + self.alpha * (target - old_val)

class MonteCarloAgent(BaseAgent):
    def __init__(self, action_space_n, gamma=0.99, epsilon=0.1):
        super().__init__(action_space_n, gamma, 0.0, epsilon)
        self.episode_history = []
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)

    def update(self, state, action, reward, next_state, done):
        self.episode_history.append((state, action, reward))
        if done:
            G = 0
            visited = set()
            for s, a, r in reversed(self.episode_history):
                G = self.gamma * G + r
                if (s, a) not in visited:
                    visited.add((s, a))
                    self.returns_sum[(s, a)] += G
                    self.returns_count[(s, a)] += 1.0
                    self.q_table[s][a] = self.returns_sum[(s, a)] / self.returns_count[(s, a)]
            self.episode_history = []

class TD0Agent(BaseAgent):
    def __init__(self, action_space_n, gamma=0.99, alpha=0.1, epsilon=0.1):
        super().__init__(action_space_n, gamma, alpha, epsilon)
        self.episode = []

    def update(self, state, action, reward, next_state, done):
        self.episode.append((state, action, reward))
        if done:
            T = len(self.episode)
            for t in range(T):
                s, a, r = self.episode[t]

                if t < T - 1:
                    next_s, next_a, _ = self.episode[t + 1]
                    target = r + self.gamma * self.q_table[next_s][next_a]
                else:
                    target = r
                
                old_val = self.q_table[s][a]
                self.q_table[s][a] = old_val + self.alpha * (target - old_val)
            self.episode = []

class NStepTDAgent(BaseAgent):
    def __init__(self, action_space_n, n_steps=3, gamma=0.99, alpha=0.1, epsilon=0.1):
        super().__init__(action_space_n, gamma, alpha, epsilon)
        self.n_steps = n_steps
        self.episode = []

    def update(self, state, action, reward, next_state, done):
        self.episode.append((state, action, reward))
        if done:
            T = len(self.episode)
            for t in range(T):

                G = 0
                limit = min(t + self.n_steps, T)
                

                for k in range(t, limit):
                    r_k = self.episode[k][2]
                    G += (self.gamma ** (k - t)) * r_k
                if t + self.n_steps < T:
                    s_n, a_n, _ = self.episode[t + self.n_steps]
                    G += (self.gamma ** self.n_steps) * self.q_table[s_n][a_n]
                
                s_t, a_t, _ = self.episode[t]
                old_val = self.q_table[s_t][a_t]
                self.q_table[s_t][a_t] = old_val + self.alpha * (G - old_val)
            self.episode = []

class PolicyEvaluationAgent(BaseAgent):
    def __init__(self, action_space_n, gamma=0.99, theta=1e-8):
        super().__init__(action_space_n, gamma)
        self.theta = theta
        self.V = defaultdict(float)

    def solve(self, env):
        P = env.unwrapped.P
        states = list(P.keys())
        
        while True:
            delta = 0
            for s in states:
                v = self.V[s]
                new_v = 0
                prob = 1.0 / self.action_space_n
                for a in range(self.action_space_n):
                    for transition_prob, next_s, r, done in P[s][a]:
                        new_v += prob * transition_prob * (r + self.gamma * self.V[next_s])
                self.V[s] = new_v
                delta = max(delta, abs(v - new_v))
            if delta < self.theta:
                break
    
    def choose_action(self, state, greedy=False):
        return random.randint(0, self.action_space_n - 1)

class PolicyIterationAgent(BaseAgent):
    def __init__(self, action_space_n, gamma=0.99, theta=1e-8):
        super().__init__(action_space_n, gamma)
        self.theta = theta
        self.V = defaultdict(float)
        self.policy = defaultdict(int)

    def solve(self, env):
        P = env.unwrapped.P
        states = list(P.keys())

        for s in states:
            self.policy[s] = random.randint(0, self.action_space_n - 1)
            
        while True:
            # 2. Policy Evaluation
            while True:
                delta = 0
                for s in states:
                    v = self.V[s]
                    a = self.policy[s]
                    new_v = 0
                    for transition_prob, next_s, r, done in P[s][a]:
                        new_v += transition_prob * (r + self.gamma * self.V[next_s])
                    self.V[s] = new_v
                    delta = max(delta, abs(v - new_v))
                if delta < self.theta:
                    break
            
            # 3. Policy Improvement
            policy_stable = True
            for s in states:
                old_action = self.policy[s]
                best_action = None
                best_val = float('-inf')
                
                for a in range(self.action_space_n):
                    val = 0
                    for transition_prob, next_s, r, done in P[s][a]:
                        val += transition_prob * (r + self.gamma * self.V[next_s])
                    if val > best_val:
                        best_val = val
                        best_action = a
                
                self.policy[s] = best_action
                if old_action != best_action:
                    policy_stable = False
            
            if policy_stable:
                break

    def choose_action(self, state, greedy=False):
        return self.policy[state]

class ValueIterationAgent(BaseAgent):
    def __init__(self, action_space_n, gamma=0.99, theta=1e-8):
        super().__init__(action_space_n, gamma)
        self.theta = theta
        self.V = defaultdict(float)
        self.policy = defaultdict(int)

    def solve(self, env):
        P = env.unwrapped.P
        states = list(P.keys())
        
        # 1. Value Iteration
        while True:
            delta = 0
            for s in states:
                v = self.V[s]
                best_val = float('-inf')
                for a in range(self.action_space_n):
                    val = 0
                    for transition_prob, next_s, r, done in P[s][a]:
                        val += transition_prob * (r + self.gamma * self.V[next_s])
                    if val > best_val: best_val = val
                
                self.V[s] = best_val
                delta = max(delta, abs(v - best_val))
            if delta < self.theta:
                break
                
        # 2. Extract Policy
        for s in states:
            best_action = None
            best_val = float('-inf')
            for a in range(self.action_space_n):
                val = 0
                for transition_prob, next_s, r, done in P[s][a]:
                    val += transition_prob * (r + self.gamma * self.V[next_s])
                if val > best_val:
                    best_val = val
                    best_action = a
            self.policy[s] = best_action

    def choose_action(self, state, greedy=False):
        return self.policy[state]

# 2. CLASSES: ENVIRONMENTS


def discretize_breakout(paddle_x, paddle_w, b_x, b_y, b_dx, b_dy):
    paddle_center = paddle_x + paddle_w / 2
    diff_x = b_x - paddle_center
    
    if diff_x < -paddle_w: bucket = 0
    elif diff_x < -paddle_w/2: bucket = 1
    elif diff_x < 0: bucket = 2
    elif diff_x < paddle_w/2: bucket = 3
    elif diff_x < paddle_w: bucket = 4
    else: bucket = 5
    
    return (bucket, int(b_dx > 0), int(b_dy > 0))

class CustomBreakout:
    def __init__(self):
        self.width = 400
        self.height = 300
        self.surface = pygame.Surface((self.width, self.height))
        
        # Game Constants
        self.paddle_w = 60
        self.paddle_h = 10
        self.ball_r = 5
        self.brick_w = 40
        self.brick_h = 15
        self.cols = self.width // self.brick_w
        self.rows = 4
        self.action_space = gym.spaces.Discrete(3)
        self.reset()

    def reset(self):
        self.paddle_x = (self.width - self.paddle_w) // 2
        self.ball_x, self.ball_y = self.width // 2, self.height // 2
        self.ball_dx = random.choice([-4, 4])
        self.ball_dy = -4
        self.score = 0
        
        # Create Bricks
        self.bricks = []
        colors = [(200, 72, 72), (200, 150, 72), (72, 200, 72), (66, 135, 245)]
        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(c * self.brick_w, r * self.brick_h + 30, self.brick_w - 2, self.brick_h - 2)
                self.bricks.append({'rect': rect, 'color': colors[r % 4], 'active': True})
                
        return self.get_state(), {}

    def get_state(self):
        return discretize_breakout(self.paddle_x, self.paddle_w, self.ball_x, self.ball_y, self.ball_dx, self.ball_dy)

    def step(self, action):
        # Paddle Movement
        if action == 1: self.paddle_x = max(0, self.paddle_x - 10)
        elif action == 2: self.paddle_x = min(self.width - self.paddle_w, self.paddle_x + 10)
        
        # Ball Movement
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Wall Collision
        if self.ball_x <= 0 or self.ball_x >= self.width: self.ball_dx *= -1
        if self.ball_y <= 0: self.ball_dy *= -1
        
        reward = 0
        done = False
        
        # Paddle Collision
        paddle_rect = pygame.Rect(self.paddle_x, self.height - 20, self.paddle_w, self.paddle_h)
        ball_rect = pygame.Rect(self.ball_x - self.ball_r, self.ball_y - self.ball_r, self.ball_r*2, self.ball_r*2)
        
        if ball_rect.colliderect(paddle_rect) and self.ball_dy > 0:
            self.ball_dy *= -1
            reward = 0.1 # Small reward for hitting paddle
            
        # Brick Collision
        hit_index = ball_rect.collidelist([b['rect'] for b in self.bricks if b['active']])
        if hit_index != -1:
            active_bricks = [b for b in self.bricks if b['active']]
            brick = active_bricks[hit_index]
            brick['active'] = False
            self.ball_dy *= -1
            reward = 1.0
            self.score += 10
            
            if all(not b['active'] for b in self.bricks):
                done = True
                reward = 10.0 # Win
        
        # Death
        if self.ball_y > self.height:
            done = True
            reward = -10
            
        return self.get_state(), reward, done, False, {}

    def render(self):
        self.surface.fill((0, 0, 0))
        
        # Draw Bricks
        for b in self.bricks:
            if b['active']:
                pygame.draw.rect(self.surface, b['color'], b['rect'])
                
        # Draw Paddle
        pygame.draw.rect(self.surface, (66, 135, 245), (self.paddle_x, self.height - 20, self.paddle_w, self.paddle_h))
        
        # Draw Ball
        pygame.draw.circle(self.surface, (255, 255, 255), (int(self.ball_x), int(self.ball_y)), self.ball_r)
        
        view = pygame.surfarray.array3d(self.surface)
        return np.transpose(view, (1, 0, 2))

class GymWrapper:
    def __init__(self, env_name):
        # FrozenLake needs render_mode to return arrays
        if "FrozenLake" in env_name:
            self.env = gym.make(env_name, render_mode="rgb_array", is_slippery=False)
        else:
            self.env = gym.make(env_name, render_mode="rgb_array")
        self.env_name = env_name
        self.action_space = self.env.action_space
        self.is_discrete_obs = isinstance(self.env.observation_space, gym.spaces.Discrete)
    
    def reset(self):
        obs, info = self.env.reset()
        if self.is_discrete_obs: return obs, info
        return self.discretize(obs), info
        
    def step(self, action):
        obs, r, term, trunc, i = self.env.step(action)
        done = term or trunc
        
        # Custom Reward for FrozenLake to encourage movement
        if r == 0 and done: r = -1 # Fell in hole
        if r == 0 and not done: r = -0.01 # Step penalty
        
        # Custom Reward for MountainCar to make it easier
        if "MountainCar" in self.env_name:
            # Reward shaping: velocity * 10 + position offset
            r += abs(obs[1]) * 10 + (obs[0] + 0.5)
        
        state = obs if self.is_discrete_obs else self.discretize(obs)
        return state, r, done, False, i
        
    def render(self):
        return self.env.render()
        
    def discretize(self, obs):
        if isinstance(obs, np.ndarray):
            if "CartPole" in self.env_name:
                bins = [np.linspace(-2.4, 2.4, 5), np.linspace(-3, 3, 5), np.linspace(-0.2, 0.2, 5), np.linspace(-2, 2, 5)]
            elif "MountainCar" in self.env_name:
                bins = [np.linspace(-1.2, 0.6, 10), np.linspace(-0.07, 0.07, 10)]
            else:
                return tuple(obs) # Fallback
                
            state = []
            for i, val in enumerate(obs):
                state.append(np.digitize(val, bins[i]))
            return tuple(state)
        return obs


# 3. STREAMLIT APP


st.set_page_config(layout="wide", page_title="RL Studio Web")

st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold;}
    .reportview-container { background: #0e1117; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.title("🎮 RL Studio Settings")
env_option = st.sidebar.selectbox("Select Environment", ["Custom Breakout", "FrozenLake-v1", "CartPole-v1", "MountainCar-v0"])
algo_option = st.sidebar.selectbox("Select Algorithm", ["Q-Learning", "SARSA", "Monte Carlo", "TD(0)", "n-step TD", "Policy Evaluation", "Policy Iteration", "Value Iteration"])

st.sidebar.subheader("Hyperparameters")
alpha = st.sidebar.slider("Learning Rate (alpha)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (gamma)", 0.1, 1.0, 0.99)
epsilon = st.sidebar.slider("Exploration Rate (epsilon)", 0.0, 1.0, 0.1)
episodes = st.sidebar.number_input("Training Episodes", 1, 5000, 500)

n_steps = 1
if algo_option == "n-step TD":
    n_steps = st.sidebar.number_input("n-steps", 1, 100, 3)

st.sidebar.subheader("Visualization")
# This checkbox replaces the automatic skipping behavior
visualize_training = st.sidebar.checkbox("Visualize Training", value=True, help="Uncheck to train instantly (Skip to end)")
speed = st.sidebar.select_slider("Animation Speed", options=["Slow", "Normal", "Fast"], value="Normal")
speed_map = {"Slow": 0.1, "Normal": 0.01, "Fast": 0.001}

# --- MAIN PAGE ---
st.title(f"Reinforcement Learning: {algo_option} on {env_option}")

col1, col2 = st.columns([2, 1])

# Session State Initialization
if 'agent' not in st.session_state: st.session_state['agent'] = None
if 'train_history' not in st.session_state: st.session_state['train_history'] = []
if 'trained' not in st.session_state: st.session_state['trained'] = False

def train_agent():
    # 1. Setup Env
    if env_option == "Custom Breakout": env = CustomBreakout()
    else: env = GymWrapper(env_option)
    
    action_n = env.action_space.n
    
    # 2. Setup Agent
    if algo_option == "Q-Learning": agent = QLearningAgent(action_n, gamma, alpha, epsilon)
    elif algo_option == "SARSA": agent = SARSAAgent(action_n, gamma, alpha, epsilon)
    elif algo_option == "Monte Carlo": agent = MonteCarloAgent(action_n, gamma, epsilon)
    elif algo_option == "TD(0)": agent = TD0Agent(action_n, gamma, alpha, epsilon)
    elif algo_option == "n-step TD": agent = NStepTDAgent(action_n, n_steps, gamma, alpha, epsilon)
    elif algo_option == "Policy Evaluation": agent = PolicyEvaluationAgent(action_n, gamma)
    elif algo_option == "Policy Iteration": agent = PolicyIterationAgent(action_n, gamma)
    elif algo_option == "Value Iteration": agent = ValueIterationAgent(action_n, gamma)
    
    # 2.5 Solver Step (Model Based)
    if algo_option in ["Policy Evaluation", "Policy Iteration", "Value Iteration"]:
        if "FrozenLake" not in env_option:
            st.error("Model-based algorithms (DP) only work with FrozenLake-v1!")
            return
        
        with st.spinner(f"Running {algo_option}..."):
             # Access inner gym env from GymWrapper
             gym_env = env.env 
             agent.solve(gym_env)
        st.success("Solved/Evaluated!")
    
    st.session_state['train_history'] = []
    
    # 3. UI Placeholders
    frame_placeholder = col1.empty()
    chart_placeholder = col2.empty()
    stats_text = col1.empty()
    progress_bar = st.progress(0)
    
    # 4. Training Loop
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        # SARSA requires initial action
        action = agent.choose_action(state)
        
        while not done:
            # RENDER LOGIC: Only render if user wants to visualize
            if visualize_training:
                frame = env.render()
                # Scale up small images for visibility
                if frame.shape[0] < 300: 
                    # Use cv2 or just simple Streamlit width argument
                    pass 
                frame_placeholder.image(frame, channels="RGB", width=500)
                time.sleep(speed_map[speed])
            
            # STEP LOGIC
            if algo_option == "SARSA":
                next_state, reward, done, _, _ = env.step(action)
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, done, next_action)
                state = next_state
                action = next_action
            else:
                if algo_option != "SARSA": action = agent.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
            
            total_reward += reward
            
        st.session_state['train_history'].append(total_reward)
        progress_bar.progress((ep + 1) / episodes)
        
        # Update Chart periodically to save performance
        if ep % 10 == 0 or ep == episodes - 1:
            df = pd.DataFrame(st.session_state['train_history'], columns=["Reward"])
            chart_placeholder.line_chart(df)
            stats_text.info(f"Training... Episode: {ep+1}/{episodes} | Last Reward: {total_reward:.2f}")

    st.session_state['agent'] = agent
    st.session_state['trained'] = True
    st.success("Training Complete!")

def test_agent():
    if not st.session_state['agent']: return
    
    if env_option == "Custom Breakout": env = CustomBreakout()
    else: env = GymWrapper(env_option)
    
    agent = st.session_state['agent']
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    frame_placeholder = col1.empty()
    st.info("Running Trained Agent...")
    
    while not done:
        frame = env.render()
        frame_placeholder.image(frame, channels="RGB", width=500)
        time.sleep(0.05) # Standard speed for watching result
        
        # Greedy action selection
        action = agent.choose_action(state, greedy=True)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state
        total_reward += reward
        
    st.success(f"Game Over! Final Score: {total_reward}")


# --- BUTTONS AREA ---
with col1:
    # Train Button
    if st.button("🚀 Start New Training"):
        train_agent()
    
    # Watch Result Button (Only appears after training)
    if st.session_state['trained']:
        st.write("---")
        if st.button("🎬 Watch Trained Agent Play"):
            test_agent()
