import env
import numpy as np

class value_iteration():
    def __init__(self, env,gamma, n_iter):
        self.env= env
        self.gamma= gamma
        self.n_iter= n_iter
        self.num_states= env.observation_space.n
        self.num_actions= env.action_space.n
        self.V= [0 for _ in range(self.num_states)]
        self.optimal_policy= [0 for _ in range(self.num_states)]

    def iterate_value(self):
        for i in range(self.n_iter):
            for s in range(self.num_states):
                q_actions = [0 for _ in range(self.num_actions)]
                for a in range(self.num_actions):
                    state_data = self.env.unwrapped.P[s][a]# prob, state, reward, done
                    prob = np.array([state_data[i][0] for i in range(len(state_data))])
                    r = np.array([state_data[i][2] for i in range(len(state_data))])
                    done = np.array([state_data[i][3] for i in range(len(state_data))])
                    next_v = np.array([self.V[state_data[i][1]] * (1 - done[i]) for i in range(len(state_data))])
                    q= sum(prob * (r + self.gamma * next_v))
                    q_actions[a]= q 
                self.V[s]= max(q_actions)
        print("Value Function after {} iterations: \n{}".format(self.n_iter, np.array(self.V)))

    def get_optimal_policy(self):
        for s in range(self.num_states):
            q_actions = [0 for _ in range(self.num_actions)]
            for a in range(self.num_actions):
                state_data = self.env.unwrapped.P[s][a]# prob, state, reward, done
                prob = np.array([state_data[i][0] for i in range(len(state_data))])
                r = np.array([state_data[i][2] for i in range(len(state_data))])
                done = np.array([state_data[i][3] for i in range(len(state_data))])
                next_v = np.array([self.V[state_data[i][1]] * (1 - done[i]) for i in range(len(state_data))])
                q= sum(prob * (r + self.gamma * next_v))
                q_actions[a]= q 
            self.optimal_policy[s]= np.argmax(q_actions)
        print("Optimal Policy: \n{}".format(np.array(self.optimal_policy)))        
        return self.optimal_policy

if __name__ == "__main__":
    env_instance = env.env
    current_state = env_instance.reset()
    vi = value_iteration(env_instance, gamma=0.9, n_iter=1000)
    vi.iterate_value()
    policy = vi.get_optimal_policy()

    s = 0
    done = False
    while not done:
        t = env_instance.step(int(policy[s]))
        s = int(t[0])
        done = t[2] or t[3]
        env_instance.render()
    env_instance.close()