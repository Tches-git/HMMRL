import numpy as np
from scipy.special import softmax

class RLOptimizer:
    def __init__(self, n_states, learning_rate=0.1, gamma=0.9):
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = np.zeros((n_states, n_states))
        self.experience_buffer = []
        
    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.n_states)
        return np.argmax(self.q_table[state])
    
    def store_experience(self, state, action, reward, next_state):
        self.experience_buffer.append((state, action, reward, next_state))
        
    def update(self, state, action, reward, next_state):
        self.store_experience(state, action, reward, next_state)
        
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value
        
        if len(self.experience_buffer) > 1000:
            self._experience_replay()
    
    def _experience_replay(self, batch_size=32):
        if len(self.experience_buffer) < batch_size:
            return
            
        batch = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        for idx in batch:
            state, action, reward, next_state = self.experience_buffer[idx]
            self.update(state, action, reward, next_state)
        
    def get_transition_matrix(self):
        transition_matrix = softmax(self.q_table, axis=1)
        return transition_matrix