from core.agent import Agent

class ValueIteration(Agent):
    
    def __init__(self, env, learning_params):
        super().__init__(env, learning_params)
        self.theta = learning_params.get('theta', 1e-6) 
        if not hasattr(env, 'get_transitions'):
            raise ValueError("Discrete Environment Only")
    
    def train(self, num_episodes=0):
        
        for s in self.env.states:
            self.V[s] = 0.0
        
        i = 0
        
        while True:
            i += 1
            delta = 0.0
            
            for s in self.env.states:
                if self.env.is_terminal_state(s):
                    self.V[s] = 0.0
                    continue
                
                old_v = self.V[s]
                best_q = float('-inf')
                
                for a in self.env.actions:
                    q_s_a = 0.0
                    transitions = self.env.get_transitions(s, a)
                    
                    for s_, r, prob in transitions:
                        q_s_a += prob * (r + self.gamma * self.V[s_])
                    
                    if q_s_a > best_q:
                        best_q = q_s_a
                
                self.V[s] = best_q
                delta = max(delta, abs(old_v - self.V[s]))
            
            self.training_data['convergence_metric'].append(i)
            
            if delta < self.theta:
                break
        
        for s in self.env.states:
            if self.env.is_terminal_state(s):
                continue
            
            best_action = None
            best_q = float('-inf')
            
            for a in self.env.actions:
                q_s_a = 0.0
                transitions = self.env.get_transitions(s, a)
                
                for s_, r, prob in transitions:
                    q_s_a += prob * (r + self.gamma * self.V[s_])
                
                if q_s_a > best_q:
                    best_q = q_s_a
                    best_action = a
            
            self.policy[s] = {best_action: 1.0}
        
        return {
            'V': self.V,
            'policy': self.policy,
            **self.training_data
        }
    
    def select_action(self, state, training=True):
        if self.env.is_terminal_state(state):
            return self.env.random_sample_action()
        
        best_action = max(self.policy[state], key=self.policy[state].get)
        return best_action