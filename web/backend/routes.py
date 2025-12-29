from flask import request, jsonify, send_from_directory
from algorithms import (PolicyIteration, ValueIteration, MonteCarlo, 
                       Sarsa, QLearning, NstepTemporalDifference)
from envs import GridWorld, FrozenLake, CartPole, MountainCar
from config import agents, next_agent_id

def register_routes(app):
    
    @app.route('/')
    def index():
        return send_from_directory('../frontend', 'index.html')
    
    @app.route('/api/environments', methods=['GET'])
    def list_environments():
        return jsonify({
            'environments': [
                {'id': 'gridworld', 'name': 'GridWorld', 'type': 'discrete'},
                {'id': 'frozenlake', 'name': 'FrozenLake', 'type': 'discrete'},
                {'id': 'cartpole', 'name': 'CartPole', 'type': 'continuous'},
                {'id': 'mountaincar', 'name': 'MountainCar', 'type': 'continuous'}
            ]
        })
    
    @app.route('/api/algorithms', methods=['GET'])
    def list_algorithms():
        return jsonify({
            'algorithms': [
                {'id': 'policy_iteration', 'name': 'Policy Iteration', 'type': 'model-based'},
                {'id': 'value_iteration', 'name': 'Value Iteration', 'type': 'model-based'},
                {'id': 'monte_carlo', 'name': 'Monte Carlo', 'type': 'model-free'},
                {'id': 'sarsa', 'name': 'SARSA', 'type': 'model-free'},
                {'id': 'qlearning', 'name': 'Q-Learning', 'type': 'model-free'},
                {'id': 'nstep_td', 'name': 'N-Step TD', 'type': 'model-free'}
            ]
        })
    
    @app.route('/api/create', methods=['POST'])
    def create_agent():
        global next_agent_id
        
        data = request.json
        env_id = data.get('environment')
        algo_id = data.get('algorithm')
        params = data.get('parameters', {})
        
        # Create environment
        if env_id == 'gridworld':
            env = GridWorld()
        elif env_id == 'frozenlake':
            env = FrozenLake()
        elif env_id == 'cartpole':
            env = CartPole(bins_per_dim=params.get('bins', 10))  
        elif env_id == 'mountaincar':
            env = MountainCar(bins_per_dim=params.get('bins', 20))  
        else:
            return jsonify({'error': 'Unknown environment'}), 400
        
        # Create learning parameters
        learning_params = {
            'gamma': params.get('gamma', 0.99),
            'alpha': params.get('alpha', 0.1),
            'epsilon': params.get('epsilon', 0.1),
            'theta': params.get('theta', 1e-6),
            'n': params.get('n', 3)  # For n-step only
        }
        
        # Create agent
        if algo_id == 'policy_iteration':
            agent = PolicyIteration(env, learning_params)
        elif algo_id == 'value_iteration':
            agent = ValueIteration(env, learning_params)
        elif algo_id == 'monte_carlo':
            mc_type = params.get('mc_type', 'first_visit')
            agent = MonteCarlo(env, learning_params, mc_type=mc_type)
        elif algo_id == 'sarsa':
            agent = Sarsa(env, learning_params)
        elif algo_id == 'qlearning':
            agent = QLearning(env, learning_params)
        elif algo_id == 'nstep_td':
            agent = NstepTemporalDifference(env, learning_params)
        else:
            return jsonify({'error': 'Unknown algorithm'}), 400
        
        # Store agent
        agent_id = f"agent_{next_agent_id}"
        next_agent_id += 1
        agents[agent_id] = {
            'agent': agent,
            'env': env,
            'env_id': env_id,
            'algo_id': algo_id
        }
        
        return jsonify({
            'agent_id': agent_id,
            'environment': env_id,
            'algorithm': algo_id
        })
    
    @app.route('/api/train', methods=['POST'])
    def train_agent():
        data = request.json
        agent_id = data.get('agent_id')
        num_episodes = data.get('num_episodes', 1000)
        
        if agent_id not in agents:
            return jsonify({'error': 'Agent not found'}), 404
        
        agent_data = agents[agent_id]
        agent = agent_data['agent']
        
        results = agent.train(num_episodes=num_episodes)
        
        # Converting results to JSON-serializable format
        json_results = {
            'episode_rewards': results.get('episode_rewards', []),
            'episode_lengths': results.get('episode_lengths', []),
            'convergence_metric': results.get('convergence_metric', [])
        }
        
        if 'V' in results:
            json_results['V'] = {str(k): float(v) for k, v in results['V'].items()}
        if 'Q' in results:
            json_results['Q'] = {str(k): float(v) for k, v in results['Q'].items()}
        if 'policy' in results:
            json_results['policy'] = {str(k): dict(v) for k, v in results['policy'].items()}
        
        return jsonify(json_results)
    
    @app.route('/api/value_function/<agent_id>', methods=['GET'])
    def get_value_function(agent_id):
        if agent_id not in agents:
            return jsonify({'error': 'Agent not found'}), 404
        
        agent = agents[agent_id]['agent']
        V = agent.get_state_value()
        
        return jsonify({
            'V': {str(k): float(v) for k, v in V.items()}
        })
    
    @app.route('/api/policy/<agent_id>', methods=['GET'])
    def get_policy(agent_id):
        if agent_id not in agents:
            return jsonify({'error': 'Agent not found'}), 404
        
        agent = agents[agent_id]['agent']
        policy = agent.get_policy()
        
        return jsonify({
            'policy': {str(k): dict(v) for k, v in policy.items()}
        })
    
    @app.route('/api/test/<agent_id>', methods=['POST'])
    def test_agent(agent_id):
        if agent_id not in agents:
            return jsonify({'error': 'Agent not found'}), 404
        
        agent = agents[agent_id]['agent']
        
        # Run test episode
        episode, states_visited, total_reward = agent.run_episode(training=False)
        
        return jsonify({
            'episode': [(str(s), int(a), float(r)) for s, a, r in episode],
            'states_visited': [str(s) for s in states_visited],
            'total_reward': float(total_reward)
        })
    
    @app.route('/api/env_info/<agent_id>', methods=['GET'])
    def get_env_info(agent_id):
        if agent_id not in agents:
            return jsonify({'error': 'Agent not found'}), 404
        
        env = agents[agent_id]['env']
        env_id = agents[agent_id]['env_id']
        
        info = {
            'env_id': env_id,
            'num_actions': int(env.num_actions),
            'actions': [int(a) for a in env.actions]
        }
        
        if env_id == 'gridworld':
            info['size'] = int(env.size)
            info['start'] = tuple(env.start)
            info['goal'] = tuple(env.goal)
        
        return jsonify(info)
    
    @app.route('/api/update_params/<agent_id>', methods=['POST'])
    def update_params(agent_id):
        if agent_id not in agents:
            return jsonify({'error': 'Agent not found'}), 404
        params = request.json
        agent = agents[agent_id]['agent']
        agent.gamma = params.get('gamma', agent.gamma)
        agent.alpha = params.get('alpha', agent.alpha)
        agent.epsilon = params.get('epsilon', agent.epsilon)
        if hasattr(agent, 'n'):
            agent.n = params.get('n', agent.n)
            
        return jsonify({'success': True})