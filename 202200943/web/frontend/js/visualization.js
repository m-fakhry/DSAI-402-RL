class Visualizer {
    constructor() {
        this.canvas = document.getElementById('env-canvas');
        this.ctx = this.canvas.getContext('2d');
    }

    // Helper: Convert state to string key (handles tuples and integers)
    stateToKey(state) {
        if (typeof state === 'string') return state;
        if (Array.isArray(state)) return `(${state.join(', ')})`;
        return String(state);
    }

    // Helper: Parse state key back to format
    parseStateKey(key) {
        // Try tuple format: "(x, y)"
        const tupleMatch = key.match(/\((\d+),\s*(\d+)\)/);
        if (tupleMatch) {
            return [parseInt(tupleMatch[1]), parseInt(tupleMatch[2])];
        }
        // Try integer
        const intVal = parseInt(key);
        if (!isNaN(intVal)) return intVal;
        return key;
    }

    // Helper: Get value from V or Q 
    getStateValue(state, V, Q) {
        const key = this.stateToKey(state);
        if (V && V[key] !== undefined) return V[key];
        if (Q) {
            
            let maxQ = -Infinity;
            for (let a = 0; a < 4; a++) { // Assuming max 4 actions
                const qKey = `(${key}, ${a})`;
                if (Q[qKey] !== undefined) {
                    maxQ = Math.max(maxQ, Q[qKey]);
                }
            }
            return maxQ !== -Infinity ? maxQ : 0;
        }
        return 0;
    }

    drawGridWorld(size, V, policy, testPath = null, Q = null) {
        const cellSize = this.canvas.width / size;
        
        // Collect all state values (from V or computed from Q)
        const allValues = [];
        for (let x = 0; x < size; x++) {
            for (let y = 0; y < size; y++) {
                const state = [x, y];
                const val = this.getStateValue(state, V, Q);
                allValues.push(val);
            }
        }
        
        const minV = allValues.length > 0 ? Math.min(...allValues) : 0;
        const maxV = allValues.length > 0 ? Math.max(...allValues) : 1;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw cells with value heatmap
        for (let x = 0; x < size; x++) {
            for (let y = 0; y < size; y++) {
                const state = [x, y];
                const val = this.getStateValue(state, V, Q);
                const norm = maxV !== minV ? (val - minV) / (maxV - minV) : 0;
                
                // Color gradient: green (low) to yellow (high)
                const r = Math.round(255 * norm);
                const g = 255;
                const b = Math.round(255 * (1 - norm));
                this.ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                this.ctx.fillRect(y * cellSize, x * cellSize, cellSize, cellSize);
                
                // Border
                this.ctx.strokeStyle = '#333';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(y * cellSize, x * cellSize, cellSize, cellSize);
                
                // Value text
                this.ctx.fillStyle = 'black';
                this.ctx.font = `${Math.max(10, cellSize * 0.15)}px Arial`;
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(val.toFixed(2), y * cellSize + cellSize / 2, x * cellSize + cellSize / 2 - 8);
            }
        }

        // Draw policy arrows
        if (policy) {
            for (let x = 0; x < size; x++) {
                for (let y = 0; y < size; y++) {
                    const state = [x, y];
                    const stateKey = this.stateToKey(state);
                    const actions = policy[stateKey];
                    if (actions && typeof actions === 'object') {
                        const action = parseInt(Object.keys(actions)[0]);
                        this.ctx.strokeStyle = '#1a4d35';
                        this.ctx.lineWidth = 2;
                        this.drawArrow(y * cellSize + cellSize / 2, x * cellSize + cellSize / 2, action, cellSize / 3);
                    }
                }
            }
        }

        // Draw test path
        if (testPath && Array.isArray(testPath) && testPath.length > 0) {
            this.ctx.strokeStyle = '#ff0000';
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            
            testPath.forEach((stateStr, i) => {
                const state = this.parseStateKey(stateStr);
                let x, y;
                if (Array.isArray(state)) {
                    [x, y] = state;
                } else {
                    return;
                }
                
                const cx = y * cellSize + cellSize / 2;
                const cy = x * cellSize + cellSize / 2;
                
                if (i === 0) {
                    this.ctx.moveTo(cx, cy);
                } else {
                    this.ctx.lineTo(cx, cy);
                }
            });
            this.ctx.stroke();
        }
    }

    drawArrow(x, y, action, length) {
        this.ctx.beginPath();
        let dx = 0, dy = 0;
        // Action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        switch (parseInt(action)) {
            case 0: dy = -length; break;  // UP
            case 1: dy = length; break;   // DOWN
            case 2: dx = -length; break;  // LEFT
            case 3: dx = length; break;   // RIGHT
        }
        
        this.ctx.moveTo(x, y);
        this.ctx.lineTo(x + dx, y + dy);
        this.ctx.stroke();

        // Arrowhead
        const headlen = Math.max(8, length * 0.3);
        const angle = Math.atan2(dy, dx);
        this.ctx.beginPath();
        this.ctx.moveTo(x + dx, y + dy);
        this.ctx.lineTo(x + dx - headlen * Math.cos(angle - Math.PI / 6), 
                        y + dy - headlen * Math.sin(angle - Math.PI / 6));
        this.ctx.moveTo(x + dx, y + dy);
        this.ctx.lineTo(x + dx - headlen * Math.cos(angle + Math.PI / 6), 
                        y + dy - headlen * Math.sin(angle + Math.PI / 6));
        this.ctx.stroke();
    }

    drawFrozenLake(V, policy, testPath = null, Q = null) {
        const size = 4;
        const cellSize = this.canvas.width / size;
        const layout = ['SFFF', 'FHFH', 'FFFH', 'HFFG'];
        
        // Collect all state values (from V or computed from Q)
        const allValues = [];
        for (let x = 0; x < size; x++) {
            for (let y = 0; y < size; y++) {
                const state = x * size + y;
                const val = this.getStateValue(state, V, Q);
                allValues.push(val);
            }
        }
        
        const minV = allValues.length > 0 ? Math.min(...allValues) : 0;
        const maxV = allValues.length > 0 ? Math.max(...allValues) : 1;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw cells
        for (let x = 0; x < size; x++) {
            for (let y = 0; y < size; y++) {
                const state = x * size + y;
                const stateKey = String(state);
                const tile = layout[x][y];
                const val = this.getStateValue(state, V, Q);
                const norm = maxV !== minV ? (val - minV) / (maxV - minV) : 0;
                
                
                let bgColor;
                if (tile === 'H') {
                    bgColor = '#505856'; // Dark gray for hole
                } else if (tile === 'G') {
                    bgColor = '#4fb38a'; // Green for goal
                } else if (tile === 'S') {
                    bgColor = '#7dd9ae'; // Light green for start
                } else {
                    // Frozen: gradient based on value
                    const r = Math.round(200 + 55 * norm);
                    const g = Math.round(240 + 15 * norm);
                    const b = Math.round(255);
                    bgColor = `rgb(${r}, ${g}, ${b})`;
                }
                
                this.ctx.fillStyle = bgColor;
                this.ctx.fillRect(y * cellSize, x * cellSize, cellSize, cellSize);
                
                this.ctx.strokeStyle = '#333';
                this.ctx.lineWidth = 1;
                this.ctx.strokeRect(y * cellSize, x * cellSize, cellSize, cellSize);
                
                // Tile label and value
                this.ctx.fillStyle = tile === 'H' ? '#fff' : '#000';
                this.ctx.font = `bold ${Math.max(12, cellSize * 0.2)}px Arial`;
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(tile, y * cellSize + cellSize / 2, x * cellSize + cellSize / 2 - 8);
                
                if (tile !== 'H' && tile !== 'G') {
                    this.ctx.font = `${Math.max(10, cellSize * 0.12)}px Arial`;
                    this.ctx.fillText(val.toFixed(2), y * cellSize + cellSize / 2, x * cellSize + cellSize / 2 + 12);
                }
            }
        }

        // Draw policy arrows (skip holes/goal)
        if (policy) {
            for (let x = 0; x < size; x++) {
                for (let y = 0; y < size; y++) {
                    const state = x * size + y;
                    const stateKey = String(state);
                    const tile = layout[x][y];
                    
                    if (tile !== 'H' && tile !== 'G') {
                        const actions = policy[stateKey];
                        if (actions && typeof actions === 'object') {
                            const action = parseInt(Object.keys(actions)[0]);
                            this.ctx.strokeStyle = '#1a4d35';
                            this.ctx.lineWidth = 2;
                            this.drawArrow(y * cellSize + cellSize / 2, x * cellSize + cellSize / 2, action, cellSize / 3);
                        }
                    }
                }
            }
        }

        // Draw test path
        if (testPath && Array.isArray(testPath) && testPath.length > 0) {
            this.ctx.strokeStyle = '#ff0000';
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            
            testPath.forEach((stateStr, i) => {
                const state = parseInt(stateStr);
                if (isNaN(state)) return;
                
                const x = Math.floor(state / size);
                const y = state % size;
                const cx = y * cellSize + cellSize / 2;
                const cy = x * cellSize + cellSize / 2;
                
                if (i === 0) {
                    this.ctx.moveTo(cx, cy);
                } else {
                    this.ctx.lineTo(cx, cy);
                }
            });
            this.ctx.stroke();
        }
    }

    drawContinuousEnv(envId, episodeRewards, V = {}, Q = {}) {
        const w = this.canvas.width;
        const h = this.canvas.height;
        this.ctx.clearRect(0, 0, w, h);
        
        // Background
        this.ctx.fillStyle = '#f5f1e8';
        this.ctx.fillRect(0, 0, w, h);
        
        // Title
        this.ctx.fillStyle = '#2d3d38';
        this.ctx.font = 'bold 24px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(`${envId.charAt(0).toUpperCase() + envId.slice(1)} Environment`, w/2, h/2 - 50);
        
        // Stats
        if (episodeRewards && episodeRewards.length > 0) {
            const avgReward = episodeRewards.slice(-100).reduce((a, b) => a + b, 0) / 
                            Math.min(100, episodeRewards.length);
            const maxReward = Math.max(...episodeRewards);
            const minReward = Math.min(...episodeRewards);
            
            this.ctx.font = '18px Arial';
            this.ctx.fillText(`Avg Reward (last 100): ${avgReward.toFixed(2)}`, w/2, h/2);
            this.ctx.fillText(`Max: ${maxReward.toFixed(2)} | Min: ${minReward.toFixed(2)}`, w/2, h/2 + 25);
        }
        
        // Value function info
        const valueCount = Object.keys(V).length || Object.keys(Q).length;
        if (valueCount > 0) {
            this.ctx.font = '14px Arial';
            this.ctx.fillText(`Learned ${valueCount} state values`, w/2, h/2 + 50);
        }
    }

    plotRewards(data) {
        if (!data || data.length === 0) return;
        
        const trace = {
            y: data,
            type: 'scatter',
            mode: 'lines',
            name: 'Reward',
            line: {
                color: '#2d7d5a',
                width: 2
            },
            fill: 'tozeroy',
            fillcolor: 'rgba(125, 217, 174, 0.2)'
        };
        
        const layout = {
            title: 'Episode Rewards',
            xaxis: { title: 'Episode' },
            yaxis: { title: 'Total Reward' },
            margin: { l: 60, r: 20, t: 40, b: 50 },
            paper_bgcolor: '#faf8f3',
            plot_bgcolor: 'white',
            showlegend: false
        };
        
        Plotly.newPlot('reward-chart', [trace], layout, {responsive: true});
    }

    plotLengths(data) {
        if (!data || data.length === 0) return;
        
        const trace = {
            y: data,
            type: 'scatter',
            mode: 'lines',
            name: 'Length',
            line: {
                color: '#4fb38a',
                width: 2
            }
        };
        
        const layout = {
            title: 'Episode Lengths',
            xaxis: { title: 'Episode' },
            yaxis: { title: 'Steps' },
            margin: { l: 60, r: 20, t: 40, b: 50 },
            paper_bgcolor: '#faf8f3',
            plot_bgcolor: 'white',
            showlegend: false
        };
        
        Plotly.newPlot('length-chart', [trace], layout, {responsive: true});
    }

    plotConvergence(data) {
        if (!data || data.length === 0) return;
        
        const trace = {
            y: data,
            type: 'scatter',
            mode: 'lines',
            name: 'Convergence',
            line: {
                color: '#1a4d35',
                width: 2
            }
        };
        
        const layout = {
            title: 'Convergence Metric',
            xaxis: { title: 'Iteration/Episode' },
            yaxis: { title: 'Convergence Value', type: 'log' },
            margin: { l: 60, r: 20, t: 40, b: 50 },
            paper_bgcolor: '#faf8f3',
            plot_bgcolor: 'white',
            showlegend: false
        };
        
        Plotly.newPlot('convergence-chart', [trace], layout, {responsive: true});
    }

    plotValueHeatmap(V, envId, size = null, Q = null) {
        // If no V values, try to compute from Q
        if ((!V || Object.keys(V).length === 0) && (!Q || Object.keys(Q).length === 0)) return;
        
        if (envId === 'gridworld') {
            const gridSize = size || 4;
            const matrix = [];
            
            for (let i = 0; i < gridSize; i++) {
                const row = [];
                for (let j = 0; j < gridSize; j++) {
                    const state = [i, j];
                    const val = this.getStateValue(state, V, Q);
                    row.push(val);
                }
                matrix.push(row);
            }
            
            const trace = {
                z: matrix,
                type: 'heatmap',
                colorscale: [
                    [0, '#faf8f3'],
                    [0.5, '#7dd9ae'],
                    [1, '#1a4d35']
                ],
                showscale: true,
                colorbar: {
                    title: 'Value'
                }
            };
            
            const layout = {
                title: 'Value Function Heatmap',
                xaxis: { title: 'Column' },
                yaxis: { title: 'Row', autorange: 'reversed' },
                margin: { l: 60, r: 80, t: 40, b: 50 },
                paper_bgcolor: '#faf8f3'
            };
            
            Plotly.newPlot('value-heatmap', [trace], layout, {responsive: true});
        } else if (envId === 'frozenlake') {
            const gridSize = 4;
            const matrix = [];
            
            for (let i = 0; i < gridSize; i++) {
                const row = [];
                for (let j = 0; j < gridSize; j++) {
                    const state = i * gridSize + j;
                    const val = this.getStateValue(state, V, Q);
                    row.push(val);
                }
                matrix.push(row);
            }
            
            const trace = {
                z: matrix,
                type: 'heatmap',
                colorscale: [
                    [0, '#505856'],
                    [0.5, '#7dd9ae'],
                    [1, '#4fb38a']
                ],
                showscale: true,
                colorbar: {
                    title: 'Value'
                }
            };
            
            const layout = {
                title: 'Value Function Heatmap',
                xaxis: { title: 'Column' },
                yaxis: { title: 'Row', autorange: 'reversed' },
                margin: { l: 60, r: 80, t: 40, b: 50 },
                paper_bgcolor: '#faf8f3'
            };
            
            Plotly.newPlot('value-heatmap', [trace], layout, {responsive: true});
        }
    }
}

const visualizer = new Visualizer();
