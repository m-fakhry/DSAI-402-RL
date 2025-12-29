const API_BASE = 'http://localhost:5000/api';

class API {
    async getEnvironments() {
        const response = await fetch(`${API_BASE}/environments`);
        return await response.json();
    }

    async getAlgorithms() {
        const response = await fetch(`${API_BASE}/algorithms`);
        return await response.json();
    }

    async createAgent(environment, algorithm, parameters) {
        const response = await fetch(`${API_BASE}/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ environment, algorithm, parameters })
        });
        return await response.json();
    }

    async trainAgent(agentId, numEpisodes) {
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                agent_id: agentId, 
                num_episodes: numEpisodes 
            })
        });
        return await response.json();
    }

    async getValueFunction(agentId) {
        const response = await fetch(`${API_BASE}/value_function/${agentId}`);
        return await response.json();
    }

    async getPolicy(agentId) {
        const response = await fetch(`${API_BASE}/policy/${agentId}`);
        return await response.json();
    }

    async testAgent(agentId) {
        const response = await fetch(`${API_BASE}/test/${agentId}`, {
            method: 'POST'
        });
        return await response.json();
    }

    async getEnvInfo(agentId) {
        const response = await fetch(`${API_BASE}/env_info/${agentId}`);
        return await response.json();
    }
}

const api = new API();