// State management
let currentAgentId = null;
let currentEnv = null;
let currentAlgo = null;
let trainingResults = null;

// Initialize UI
document.addEventListener('DOMContentLoaded', async () => {
    await initializeUI();
    setupEventListeners();
    initializeSliders();
});

async function initializeUI() {
    try {
        // Load environments from API
        const envs = await api.getEnvironments();
        const envSelect = document.getElementById('environment-select');
        envSelect.innerHTML = '<option value="">Select environment...</option>';
        envs.environments.forEach(env => {
            const option = document.createElement('option');
            option.value = env.id;
            option.textContent = env.name;
            envSelect.appendChild(option);
        });
        
        // Load algorithms from API
        const algos = await api.getAlgorithms();
        const algoSelect = document.getElementById('algorithm-select');
        algoSelect.innerHTML = '<option value="">Select algorithm...</option>';
        algos.algorithms.forEach(algo => {
            const option = document.createElement('option');
            option.value = algo.id;
            option.textContent = algo.name;
            algoSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to initialize UI:', error);
        updateStatus('Failed to load environments and algorithms');
    }
}

function setupEventListeners() {
    // Algorithm selection
    document.getElementById('algorithm-select').addEventListener('change', (e) => {
        currentAlgo = e.target.value;
        updateParameterVisibility();
    });
    
    // Environment selection
    document.getElementById('environment-select').addEventListener('change', (e) => {
        currentEnv = e.target.value;
    });
    
    // Buttons
    document.getElementById('train-btn').addEventListener('click', trainAgent);
    document.getElementById('test-btn').addEventListener('click', testAgent);
    document.getElementById('reset-btn').addEventListener('click', resetAll);
}

function initializeSliders() {
    const sliders = ['gamma', 'alpha', 'epsilon', 'episodes', 'nsteps'];
    
    sliders.forEach(id => {
        const slider = document.getElementById(id);
        if (slider) {
            slider.addEventListener('input', (e) => {
                document.getElementById(`${id}-value`).textContent = e.target.value;
            });
        }
    });
}

function updateParameterVisibility() {
    const mcGroup = document.getElementById('mc-type-group');
    const nstepGroup = document.getElementById('nstep-group');
    
    mcGroup.style.display = currentAlgo === 'monte_carlo' ? 'block' : 'none';
    nstepGroup.style.display = currentAlgo === 'nstep_td' ? 'block' : 'none';
}

function updateStatus(message) {
    document.getElementById('status-text').textContent = message;
}

async function trainAgent() {
    if (!currentEnv || !currentAlgo) {
        updateStatus('Please select environment and algorithm');
        return;
    }
    
    try {
        updateStatus('Creating agent...');
        document.getElementById('train-btn').disabled = true;
        
        // Gather parameters
        const parameters = {
            gamma: parseFloat(document.getElementById('gamma').value),
            alpha: parseFloat(document.getElementById('alpha').value),
            epsilon: parseFloat(document.getElementById('epsilon').value)
        };
        
        if (currentAlgo === 'monte_carlo') {
            parameters.mc_type = document.getElementById('mc-type').value;
        }
        
        if (currentAlgo === 'nstep_td') {
            parameters.n = parseInt(document.getElementById('nsteps').value);
        }
        
        // Create agent
        const createResult = await api.createAgent(currentEnv, currentAlgo, parameters);
        currentAgentId = createResult.agent_id;
        
        updateStatus('Training agent...');
        
        // Train agent
        const numEpisodes = parseInt(document.getElementById('episodes').value);
        const results = await api.trainAgent(currentAgentId, numEpisodes);
        trainingResults = results;
        
        updateStatus('Training complete! Visualizing results...');
        
        // Visualize results
        await visualizeResults(results);
        
        document.getElementById('test-btn').disabled = false;
        updateStatus('Training complete. Ready to test agent.');
        
    } catch (error) {
        console.error('Training error:', error);
        updateStatus('Error during training: ' + error.message);
    } finally {
        document.getElementById('train-btn').disabled = false;
    }
}

async function visualizeResults(results) {
    // Plot charts (algorithm-agnostic metrics)
    if (results.episode_rewards && results.episode_rewards.length > 0) {
        visualizer.plotRewards(results.episode_rewards);
    }
    
    if (results.episode_lengths && results.episode_lengths.length > 0) {
        visualizer.plotLengths(results.episode_lengths);
    }
    
    if (results.convergence_metric && results.convergence_metric.length > 0) {
        visualizer.plotConvergence(results.convergence_metric);
    }
    
    // Get environment info
    const envInfo = await api.getEnvInfo(currentAgentId);
    
    // Fetch value function and policy from API (or use from results if available)
    let V = results.V || {};
    let policy = results.policy || {};
    
    // Try to get from API if not in results
    if (Object.keys(V).length === 0) {
        try {
            const vData = await api.getValueFunction(currentAgentId);
            V = vData.V || {};
        } catch (error) {
            console.warn('Could not fetch value function:', error);
        }
    }
    
    if (Object.keys(policy).length === 0) {
        try {
            const policyData = await api.getPolicy(currentAgentId);
            policy = policyData.policy || {};
        } catch (error) {
            console.warn('Could not fetch policy:', error);
        }
    }
    
    // Algorithm-aware visualization
    // Model-based algorithms (Policy/Value Iteration) have V
    // Model-free algorithms (SARSA, Q-Learning, etc.) have Q
    const Q = results.Q || {};
    
    // Draw environment visualization
    if (currentEnv === 'gridworld') {
        const size = envInfo.size || 4;
        visualizer.drawGridWorld(size, V, policy, null, Q);
        // Show heatmap if we have V values or Q values
        if (Object.keys(V).length > 0 || Object.keys(Q).length > 0) {
            visualizer.plotValueHeatmap(V, currentEnv, size, Q);
        }
    } else if (currentEnv === 'frozenlake') {
        visualizer.drawFrozenLake(V, policy, null, Q);
        // Show heatmap if we have V values or Q values
        if (Object.keys(V).length > 0 || Object.keys(Q).length > 0) {
            visualizer.plotValueHeatmap(V, currentEnv, null, Q);
        }
    } else {
        // Continuous environments (CartPole, MountainCar)
        visualizer.drawContinuousEnv(currentEnv, results.episode_rewards, V, Q);
    }
}

async function testAgent() {
    if (!currentAgentId) {
        updateStatus('No trained agent available');
        return;
    }
    
    try {
        updateStatus('Running test episode...');
        
        const testResult = await api.testAgent(currentAgentId);
        
        updateStatus(`Test complete! Total reward: ${testResult.total_reward.toFixed(2)}`);
        
        // Visualize test path
        const envInfo = await api.getEnvInfo(currentAgentId);
        
        // Fetch current value function and policy from API
        let V = {};
        let policy = {};
        try {
            const vData = await api.getValueFunction(currentAgentId);
            V = vData.V || {};
        } catch (error) {
            console.warn('Could not fetch value function:', error);
            V = trainingResults.V || {};
        }
        
        try {
            const policyData = await api.getPolicy(currentAgentId);
            policy = policyData.policy || {};
        } catch (error) {
            console.warn('Could not fetch policy:', error);
            policy = trainingResults.policy || {};
        }
        
        if (currentEnv === 'gridworld') {
            const size = envInfo.size || 4;
            const Q = trainingResults?.Q || {};
            visualizer.drawGridWorld(size, V, policy, testResult.states_visited, Q);
        } else if (currentEnv === 'frozenlake') {
            const Q = trainingResults?.Q || {};
            visualizer.drawFrozenLake(V, policy, testResult.states_visited, Q);
        }
        
    } catch (error) {
        console.error('Test error:', error);
        updateStatus('Error during testing: ' + error.message);
    }
}

function resetAll() {
    currentAgentId = null;
    currentEnv = null;
    currentAlgo = null;
    trainingResults = null;
    
    document.getElementById('environment-select').value = '';
    document.getElementById('algorithm-select').value = '';
    document.getElementById('test-btn').disabled = true;
    
    // Clear visualizations
    visualizer.ctx.clearRect(0, 0, visualizer.canvas.width, visualizer.canvas.height);
    Plotly.purge('reward-chart');
    Plotly.purge('length-chart');
    Plotly.purge('convergence-chart');
    Plotly.purge('value-heatmap');
    
    updateStatus('Reset complete. Select environment and algorithm to begin');
}