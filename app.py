import streamlit as st
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap

# Page configuration
st.set_page_config(
    page_title="RL Algorithm Visualizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# ==================== ENVIRONMENT SETUP ====================
@st.cache_resource
def create_environment(env_name, is_slippery=False):
    """Create and cache environment"""
    if env_name == "FrozenLake-v1":
        return gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=is_slippery)
    elif env_name == "Taxi-v3":
        return gym.make("Taxi-v3", render_mode="rgb_array")
    elif env_name == "CliffWalking-v1":
        return gym.make("CliffWalking-v1", render_mode="rgb_array")
    return None

# ==================== IMPORT ALGORITHM CLASSES ====================
# Import from your existing files
from value_iteration import value_iteration
from policy_iteration import policy_iteration
from monte_carlo import monte_carlo

# Note: Implement these algorithms in separate files and import them here
# from temporal_difference import temporal_difference
# from n_step_td import n_step_td
# from sarsa import sarsa
# from q_learning import q_learning

# ==================== VISUALIZATION FUNCTIONS ====================
def plot_value_function(V, env_name, grid_size=None):
    """Plot the value function as a heatmap"""
    fig, ax = plt.subplots(figsize=(4, 3))
    
    if env_name == "FrozenLake-v1":
        grid_size = (4, 4)
    elif env_name == "CliffWalking-v1":
        grid_size = (4, 12)
    elif env_name == "Taxi-v3":
        # Taxi has 500 states, show a subset
        V_subset = V[:25]
        grid_size = (5, 5)
        V = V_subset
    
    V_grid = np.array(V).reshape(grid_size)
    
    cmap = LinearSegmentedColormap.from_list('custom', ['#ff6b6b', '#ffd93d', '#6bcb77'])
    im = ax.imshow(V_grid, cmap=cmap)
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            text = ax.text(j, i, f'{V_grid[i, j]:.2f}', ha='center', va='center', color='black', fontsize=7)
    
    ax.set_title('Value Function', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, label='State Value')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    return fig


def plot_policy(policy, env_name, grid_size=None):
    """Plot the policy with arrows"""
    fig, ax = plt.subplots(figsize=(3, 2.5))
    
    # Action mappings for different environments
    if env_name == "FrozenLake-v1":
        grid_size = (4, 4)
        action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    elif env_name == "CliffWalking-v1":
        grid_size = (4, 12)
        action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    elif env_name == "Taxi-v3":
        policy = policy[:25]
        grid_size = (5, 5)
        action_symbols = {0: '↓', 1: '↑', 2: '→', 3: '←', 4: 'P', 5: 'D'}
    
    policy_grid = np.array(policy).reshape(grid_size)
    
    ax.set_xlim(-0.5, grid_size[1] - 0.5)
    ax.set_ylim(-0.5, grid_size[0] - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            action = policy_grid[i, j]
            symbol = action_symbols.get(action, '?')
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2))
            ax.text(j, i, symbol, ha='center', va='center', fontsize=14, fontweight='bold', color='#1565c0')
    
    ax.set_title('Optimal Policy', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    plt.tight_layout()
    
    return fig


def plot_convergence(history, algorithm_type):
    """Plot convergence metrics"""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    
    if algorithm_type in ['Value Iteration']:
        iterations = [h['iteration'] for h in history]
        max_diffs = [h['max_diff'] for h in history]
        ax.plot(iterations, max_diffs, 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Max Value Change', fontsize=10)
        ax.set_title('Convergence: Max Value Difference per Iteration', fontsize=11, fontweight='bold')
        ax.set_yscale('log')
    elif algorithm_type in ['Monte Carlo', 'SARSA', 'Q-Learning', 'TD', 'n-step TD']:
        if history:
            episodes = [h['episode'] for h in history]
            epsilons = [h['epsilon'] for h in history]
            ax.plot(episodes, epsilons, 'g-', linewidth=2)
            ax.set_xlabel('Episode', fontsize=10)
            ax.set_ylabel('Epsilon', fontsize=10)
            ax.set_title('Exploration Rate Decay', fontsize=11, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    return fig


def run_inference(env, policy, max_steps=100):
    """Run inference and return trajectory with rendered frames"""
    state = env.reset()[0]
    trajectory = [state]
    rewards = []
    frames = []
    done = False
    steps = 0
    
    # Capture initial frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    
    while not done and steps < max_steps:
        action = int(policy[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory.append(next_state)
        rewards.append(reward)
        state = next_state
        done = terminated or truncated
        steps += 1
        
        # Capture frame after each step
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    
    return trajectory, rewards, done, frames


# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<p class="main-header">🤖 Reinforcement Learning Algorithm Visualizer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore and visualize different RL algorithms in action</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Environment Selection
        st.subheader("🌍 Environment")
        env_name = st.selectbox(
            "Select Environment",
            ["FrozenLake-v1", "Taxi-v3", "CliffWalking-v1"],
            help="Choose the environment for training. FrozenLake is a simple grid world, Taxi involves pickup/dropoff, CliffWalking has a dangerous cliff."
        )
        
        if env_name == "FrozenLake-v1":
            is_slippery = st.checkbox(
                "Slippery Ice",
                value=False,
                help="When enabled, the agent may slip and move in unintended directions (stochastic environment)."
            )
        else:
            is_slippery = False
        
        st.divider()
        
        # Algorithm Selection
        st.subheader("🧠 Algorithm")
        
        available_algorithms = ["Value Iteration", "Policy Iteration", "Monte Carlo", "Temporal Difference (TD)", "n-step TD", "SARSA", "Q-Learning"]
        
        algorithm = st.selectbox(
            "Select Algorithm",
            available_algorithms,
            help="Choose the RL algorithm to train and visualize."
        )
        
        st.divider()
        
        # Parameters
        st.subheader("🎛️ Parameters")
        
        gamma = st.slider(
            "Discount Factor (γ)",
            min_value=0.0,
            max_value=1.0,
            value=0.99,
            step=0.01,
            help="Determines the importance of future rewards. γ=0 means only immediate rewards matter, γ=1 means future rewards are as important as immediate ones."
        )
        
        if algorithm in ["Value Iteration", "Policy Iteration"]:
            n_iter = st.slider(
                "Max Iterations",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Maximum number of iterations for the algorithm."
            )
        else:
            n_episodes = st.slider(
                "Number of Episodes",
                min_value=100,
                max_value=50000,
                value=10000,
                step=100,
                help="Number of episodes to train the agent."
            )
        
        if algorithm in ["Monte Carlo", "Temporal Difference (TD)", "n-step TD", "SARSA", "Q-Learning"]:
            epsilon = st.slider(
                "Initial Exploration Rate (ε)",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                help="Probability of taking a random action (exploration vs exploitation). Higher values mean more exploration."
            )
            
            epsilon_decay = st.slider(
                "Epsilon Decay Rate",
                min_value=0.9,
                max_value=0.9999,
                value=0.995,
                step=0.001,
                format="%.4f",
                help="Rate at which epsilon decreases after each episode."
            )
            
            min_epsilon = st.slider(
                "Minimum Epsilon",
                min_value=0.0,
                max_value=0.5,
                value=0.01,
                step=0.01,
                help="Minimum exploration rate to maintain some exploration."
            )
        
        if algorithm in ["Temporal Difference (TD)", "n-step TD", "SARSA", "Q-Learning"]:
            alpha = st.slider(
                "Learning Rate (α)",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Step size for updating value estimates. Higher values mean faster but potentially unstable learning."
            )
        
        if algorithm == "n-step TD":
            n_steps = st.slider(
                "Number of Steps (n)",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help="Number of steps to look ahead before updating values. n=1 is equivalent to TD(0)."
            )
        
        st.divider()
        
        # Training button
        train_button = st.button("🚀 Train Agent", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns(2)
    
    # Initialize session state
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'policy' not in st.session_state:
        st.session_state.policy = None
    if 'V' not in st.session_state:
        st.session_state.V = None
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Algorithm descriptions
    algorithm_info = {
        "Value Iteration": {
            "description": "Value Iteration is a dynamic programming algorithm that iteratively computes the optimal value function by applying the Bellman optimality equation until convergence.",
            "formula": "V(s) = max_a Σ P(s'|s,a) [R(s,a,s') + γV(s')]",
            "pros": ["Guaranteed to converge to optimal policy", "Works with full environment model"],
            "cons": ["Requires complete environment model", "Computationally expensive for large state spaces"]
        },
        "Policy Iteration": {
            "description": "Policy Iteration alternates between policy evaluation (computing V for current policy) and policy improvement (updating policy greedily based on V).",
            "formula": "V^π(s) = Σ P(s'|s,π(s)) [R + γV^π(s')]",
            "pros": ["Often converges in fewer iterations than Value Iteration", "Guaranteed optimal policy"],
            "cons": ["Each iteration is more expensive", "Requires complete environment model"]
        },
        "Monte Carlo": {
            "description": "Monte Carlo methods learn from complete episodes of experience. They estimate value functions by averaging returns observed after visiting states.",
            "formula": "V(s) = average(Returns after visiting s)",
            "pros": ["Model-free - learns from experience", "Unbiased estimates"],
            "cons": ["High variance", "Must wait for episode to complete", "Only works for episodic tasks"]
        },
        "Temporal Difference (TD)": {
            "description": "TD learning combines ideas from Monte Carlo and dynamic programming. It updates estimates based on other estimates (bootstrapping) without waiting for final outcome.",
            "formula": "V(s) ← V(s) + α[R + γV(s') - V(s)]",
            "pros": ["Model-free", "Can learn before episode ends", "Lower variance than MC"],
            "cons": ["Biased estimates", "Sensitive to learning rate"]
        },
        "n-step TD": {
            "description": "n-step TD bridges the gap between TD(0) and Monte Carlo by using n steps of actual rewards before bootstrapping.",
            "formula": "G_t:t+n = R_t+1 + γR_t+2 + ... + γ^(n-1)R_t+n + γ^n V(S_t+n)",
            "pros": ["Flexible trade-off between bias and variance", "Often faster learning than TD(0)"],
            "cons": ["More complex to implement", "Requires tuning n"]
        },
        "SARSA": {
            "description": "SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm that learns Q-values while following an ε-greedy policy.",
            "formula": "Q(s,a) ← Q(s,a) + α[R + γQ(s',a') - Q(s,a)]",
            "pros": ["On-policy - learns about the policy being followed", "Safer exploration in dangerous environments"],
            "cons": ["May not find optimal policy if exploration is needed", "Slower convergence"]
        },
        "Q-Learning": {
            "description": "Q-Learning is an off-policy TD control algorithm that learns the optimal Q-function regardless of the policy being followed.",
            "formula": "Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]",
            "pros": ["Off-policy - can learn optimal policy while exploring", "Simple and widely applicable"],
            "cons": ["Can be unstable with function approximation", "May overestimate values"]
        }
    }
    
    # Display algorithm info
    with st.expander("📖 Algorithm Information", expanded=True):
        info = algorithm_info[algorithm]
        st.markdown(f"**{algorithm}**")
        st.write(info["description"])
        st.latex(info["formula"])
        
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**✅ Advantages:**")
            for pro in info["pros"]:
                st.markdown(f"- {pro}")
        with col_cons:
            st.markdown("**❌ Disadvantages:**")
            for con in info["cons"]:
                st.markdown(f"- {con}")
    
    # Training
    if train_button:
        env = create_environment(env_name, is_slippery)
        
        with st.spinner(f"Training {algorithm}..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if algorithm == "Value Iteration":
                agent = value_iteration(env, gamma, n_iter)
                agent.iterate_value()
                policy = agent.get_optimal_policy()
                st.session_state.V = np.array(agent.V)
                st.session_state.policy = np.array(policy)
                st.session_state.trained = True
                st.session_state.algorithm = algorithm
                st.session_state.env_name = env_name
            
            elif algorithm == "Policy Iteration":
                agent = policy_iteration(env, gamma, n_iter)
                policy = agent.get_optimal_policy()
                st.session_state.V = np.array(agent.V)
                st.session_state.policy = np.array(policy)
                st.session_state.trained = True
                st.session_state.algorithm = algorithm
                st.session_state.env_name = env_name
            
            elif algorithm == "Monte Carlo":
                env_train = gym.make(env_name, is_slippery=is_slippery) if env_name == "FrozenLake-v1" else gym.make(env_name)
                agent = monte_carlo(env_train, n_episodes, epsilon, epsilon_decay, min_epsilon, gamma)
                policy = agent.get_policy_epsilon()
                st.session_state.V = np.max(agent.Q, axis=1)
                st.session_state.policy = np.array(policy)
                st.session_state.trained = True
                st.session_state.algorithm = algorithm
                st.session_state.env_name = env_name
            
            elif algorithm == "Temporal Difference (TD)":
                st.warning("TD algorithm not yet implemented. Create temporal_difference.py and import it.")
                st.session_state.trained = False
            
            elif algorithm == "n-step TD":
                st.warning("n-step TD algorithm not yet implemented. Create n_step_td.py and import it.")
                st.session_state.trained = False
            
            elif algorithm == "SARSA":
                st.warning("SARSA algorithm not yet implemented. Create sarsa.py and import it.")
                st.session_state.trained = False
            
            elif algorithm == "Q-Learning":
                st.warning("Q-Learning algorithm not yet implemented. Create q_learning.py and import it.")
                st.session_state.trained = False
            
            progress_bar.progress(1.0)
            status_text.text("Training complete!")
    
    # Display results
    if st.session_state.trained:
        st.success("✅ Training completed successfully!")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Algorithm", st.session_state.algorithm)
        with col2:
            st.metric("Environment", st.session_state.env_name)
        with col3:
            if st.session_state.history:
                st.metric("Iterations/Episodes", len(st.session_state.history))
        
        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Value Function", "🎯 Policy", "📈 Convergence", "🎮 Inference"])
        
        with tab1:
            st.subheader("Value Function Heatmap")
            st.write("The value function shows the expected return from each state following the optimal policy.")
            fig_value = plot_value_function(st.session_state.V, st.session_state.env_name)
            st.pyplot(fig_value)
        
        with tab2:
            st.subheader("Optimal Policy")
            st.write("Arrows indicate the best action to take in each state.")
            fig_policy = plot_policy(st.session_state.policy, st.session_state.env_name)
            st.pyplot(fig_policy)
        
        with tab3:
            st.subheader("Training Convergence")
            if st.session_state.history:
                fig_conv = plot_convergence(st.session_state.history, st.session_state.algorithm)
                st.pyplot(fig_conv)
            else:
                st.info("Convergence data not available for this algorithm.")
        
        with tab4:
            st.subheader("Agent Inference")
            st.write("Watch the trained agent navigate the environment!")
            
            # Initialize inference session state
            if 'inference_frames' not in st.session_state:
                st.session_state.inference_frames = []
            if 'inference_trajectory' not in st.session_state:
                st.session_state.inference_trajectory = []
            if 'inference_rewards' not in st.session_state:
                st.session_state.inference_rewards = []
            if 'inference_done' not in st.session_state:
                st.session_state.inference_done = False
            
            # Speed control
            speed = st.slider("Animation Speed (seconds per step)", 0.1, 2.0, 0.5, 0.1, key="speed_slider")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                run_inference_btn = st.button("🎬 Run Inference", type="secondary", use_container_width=True)
            with col_btn2:
                clear_btn = st.button("🗑️ Clear Results", use_container_width=True)
            
            if clear_btn:
                st.session_state.inference_frames = []
                st.session_state.inference_trajectory = []
                st.session_state.inference_rewards = []
                st.session_state.inference_done = False
            
            if run_inference_btn:
                # Run inference and store results
                env_infer = gym.make(st.session_state.env_name, render_mode="rgb_array", is_slippery=False) if st.session_state.env_name == "FrozenLake-v1" else gym.make(st.session_state.env_name, render_mode="rgb_array")
                
                state = env_infer.reset()[0]
                frames = []
                trajectory = [state]
                rewards = []
                done = False
                step = 0
                max_steps = 100
                
                # Capture initial frame
                frame = env_infer.render()
                if frame is not None:
                    frames.append(frame)
                
                while not done and step < max_steps:
                    action = int(st.session_state.policy[state])
                    next_state, reward, terminated, truncated, _ = env_infer.step(action)
                    trajectory.append(next_state)
                    rewards.append(reward)
                    done = terminated or truncated
                    step += 1
                    
                    frame = env_infer.render()
                    if frame is not None:
                        frames.append(frame)
                    
                    state = next_state
                
                env_infer.close()
                
                # Store in session state
                st.session_state.inference_frames = frames
                st.session_state.inference_trajectory = trajectory
                st.session_state.inference_rewards = rewards
                st.session_state.inference_done = True
            
            # Display results if available
            if st.session_state.inference_done and st.session_state.inference_frames:
                frames = st.session_state.inference_frames
                trajectory = st.session_state.inference_trajectory
                rewards = st.session_state.inference_rewards
                
                # Action names
                if st.session_state.env_name == "FrozenLake-v1":
                    action_names = {0: '← Left', 1: '↓ Down', 2: '→ Right', 3: '↑ Up'}
                elif st.session_state.env_name == "CliffWalking-v1":
                    action_names = {0: '↑ Up', 1: '→ Right', 2: '↓ Down', 3: '← Left'}
                else:
                    action_names = {0: '↓ South', 1: '↑ North', 2: '→ East', 3: '← West', 4: 'P Pickup', 5: 'D Dropoff'}
                
                # Show policy
                st.markdown("### Policy Used")
                fig_policy_infer = plot_policy(st.session_state.policy, st.session_state.env_name)
                st.pyplot(fig_policy_infer)
                
                # Animation section
                st.markdown("### Environment Animation")
                
                # Play animation button
                if st.button("▶️ Play Animation", use_container_width=True):
                    frame_placeholder = st.empty()
                    info_placeholder = st.empty()
                    
                    for idx, frame in enumerate(frames):
                        frame_placeholder.image(frame, caption=f"Step {idx}/{len(frames)-1}", width=300)
                        if idx < len(trajectory) and idx < len(rewards) + 1:
                            if idx == 0:
                                info_placeholder.info(f"🏁 Starting at state {trajectory[0]}")
                            elif idx <= len(rewards):
                                s = trajectory[idx - 1]
                                a = int(st.session_state.policy[s])
                                info_placeholder.info(f"State: {trajectory[idx]} | Action: {action_names.get(a, str(a))} | Reward: {rewards[idx-1]}")
                        time.sleep(speed)
                    
                    if sum(rewards) > 0:
                        info_placeholder.success(f"🎉 Goal reached! Total reward: {sum(rewards)}")
                    else:
                        info_placeholder.error(f"💀 Episode ended. Total reward: {sum(rewards)}")
                
                # Manual step-through with slider
                st.markdown("### Step Through Frames")
                frame_idx = st.slider("Frame", 0, len(frames) - 1, 0, key="frame_nav_slider")
                
                col_frame, col_info = st.columns([2, 1])
                with col_frame:
                    st.image(frames[frame_idx], caption=f"Step {frame_idx}", width=300)
                
                with col_info:
                    if frame_idx == 0:
                        st.info("🏁 Start")
                        st.metric("State", trajectory[0])
                    elif frame_idx < len(trajectory):
                        st.metric("State", trajectory[frame_idx])
                        if frame_idx <= len(rewards):
                            prev_state = trajectory[frame_idx - 1]
                            action = int(st.session_state.policy[prev_state])
                            st.metric("Action", action_names.get(action, str(action)))
                            st.metric("Reward", rewards[frame_idx - 1])
                    
                    if frame_idx == len(frames) - 1:
                        if sum(rewards) > 0:
                            st.success("🎉 Goal!")
                        else:
                            st.error("💀 End")
                
                # Trajectory summary
                st.markdown("### Trajectory Summary")
                st.metric("Total Steps", len(trajectory) - 1)
                st.metric("Total Reward", sum(rewards))
                st.metric("Goal Reached", "✅ Yes" if sum(rewards) > 0 else "❌ No")
                
                # Step-by-step table
                with st.expander("📋 Step-by-Step Details", expanded=False):
                    steps_data = []
                    for i in range(len(trajectory) - 1):
                        s = trajectory[i]
                        a = int(st.session_state.policy[s])
                        ns = trajectory[i + 1]
                        r = rewards[i]
                        steps_data.append({
                            "Step": i + 1,
                            "State": s,
                            "Action": action_names.get(a, str(a)),
                            "Next State": ns,
                            "Reward": r
                        })
                    
                    if steps_data:
                        st.dataframe(steps_data, use_container_width=True, hide_index=True)
            else:
                st.info("👆 Click 'Run Inference' to see the trained agent in action!")
    
    else:
        st.info("👈 Configure parameters in the sidebar and click 'Train Agent' to start training.")
        
        # Show environment preview
        st.subheader("🌍 Environment Preview")
        env_descriptions = {
            "FrozenLake-v1": "Navigate a frozen lake from start (S) to goal (G) while avoiding holes (H). The agent can move in 4 directions.",
            "Taxi-v3": "A taxi must pickup a passenger and drop them at the destination. Actions: move in 4 directions, pickup, dropoff.",
            "CliffWalking-v1": "Navigate from start to goal while avoiding the cliff. Falling off the cliff gives a large negative reward."
        }
        st.write(env_descriptions.get(env_name, ""))
        
        # Show a simple grid preview
        env_preview = create_environment(env_name, is_slippery if env_name == "FrozenLake-v1" else False)
        if env_preview:
            env_preview.reset()
            img = env_preview.render()
            if img is not None:
                st.image(img, caption=f"{env_name} Environment", width=300)


if __name__ == "__main__":
    main()
