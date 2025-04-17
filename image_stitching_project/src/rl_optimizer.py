import torch
from scipy.special import softmax

class RLOptimizer:
    def __init__(self, n_states, learning_rate=0.1, gamma=0.9, device=torch.device("cpu")):
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = torch.zeros((n_states, n_states), device=device)
        self.experience_buffer = []
        self.device = device
        
    def get_action(self, state, epsilon=0.1):
        if torch.rand(1, device=self.device).item() < epsilon:
            return torch.randint(0, self.n_states, (1,), device=self.device).item()
        return torch.argmax(self.q_table[state]).item()
    
    def store_experience(self, state, action, reward, next_state):
        self.experience_buffer.append((state, action, reward, next_state))
        
    def update(self, state, action, reward, next_state):
        self.store_experience(state, action, reward, next_state)
        
        old_value = self.q_table[state, action]
        next_max = torch.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + \
                   self.learning_rate * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value
        
        if len(self.experience_buffer) > 1000:
            self._experience_replay()
    
    def _experience_replay(self, batch_size=32):
        if len(self.experience_buffer) < batch_size:
            return
            
        indices = torch.randint(0, len(self.experience_buffer), (batch_size,), device=self.device)
        for idx in indices:
            state, action, reward, next_state = self.experience_buffer[idx.item()]
            self.update(state, action, reward, next_state)
        
    def get_transition_matrix(self):
        transition_matrix = softmax(self.q_table.cpu().numpy(), axis=1)
        return transition_matrix