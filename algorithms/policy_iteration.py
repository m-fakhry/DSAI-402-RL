from core.agent import Agent

class PolicyIteration(Agent):
    
    def __init__(self, env, learning_params):
        super().__init__(env, learning_params)
        self.theta = learning_params.get('theta', 1e-6)  
        if not hasattr(env, 'get_transitions'):
            raise ValueError("Discrete Environment Only")
        
    def _policy_evaluation(self):
        
        while True:
            delta = 0.0
            
            for s in self.env.states:
                if self.env.is_terminal_state(s):
                    self.V[s] = 0.0
                    continue
                    
                
                
                old_v = self.V[s]
                new_v = 0.0
                for a, pi_s_a in self.policy[s].items():
                    if pi_s_a == 0:
                        continue
                            
                    q_s_a = 0.0
                    transitions = self.env.get_transitions(s, a)
                    for s_, r, prob in transitions:
                        q_s_a += prob * (r + self.gamma * self.V[s_])
                        
                    new_v += pi_s_a * q_s_a
                
                self.V[s] = new_v
                delta = max(delta, abs(old_v - new_v))  
            
            if delta < self.theta:
                break
            
    def _policy_improvement(self):
        
        policy_is_stable = True
        
        for s in self.env.states:
            if self.env.is_terminal_state(s):
                continue
            
            old_action = max(self.policy[s], key=self.policy[s].get)
            
            for a in self.env.actions:
                q_s_a = 0.0
                transitions = self.env.get_transitions(s, a)
                for s_, r, prob in transitions:
                    q_s_a += prob * (r + self.gamma * self.V[s_])
                self.Q[(s, a)] = q_s_a
                                    
            best_action = max(self.env.actions, key=lambda a: self.Q.get((s, a), 0.0))
            
            self.policy[s] = {best_action: 1.0}
            if best_action != old_action:
                policy_is_stable = False
                
        return policy_is_stable               

    def train(self, num_episodes=0):
        
        for s in self.env.states:
            self.V[s] = 0.0
        
        for s in self.env.states:
            if self.env.is_terminal_state(s) == False:
                random_action = self.env.random_sample_action()
                self.policy[s] = {random_action: 1.0}
        
        i = 0
        while True:
            i += 1
            
            self._policy_evaluation()
            policy_is_stable = self._policy_improvement()
            self.training_data['convergence_metric'].append(i)
            
            if policy_is_stable:
                break
            
        return {'V': self.V, 'policy': self.policy, **self.training_data}
    
    def select_action(self, state, training=True):
        if self.env.is_terminal_state(state):
            return self.env.random_sample_action()
        
        best_action = max(self.policy[state], key=self.policy[state].get)
        return best_action