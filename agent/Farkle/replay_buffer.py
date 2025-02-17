import numpy as np


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)
        self.action_mask_memory = np.zeros((self.mem_size, n_actions),
                                           dtype=np.bool)
        self.next_action_mask_memory = np.zeros((self.mem_size, n_actions),
                                                dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done, action_mask, next_action_mask):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.action_mask_memory[index] = action_mask
        self.next_action_mask_memory[index] = next_action_mask
        self.mem_counter += 1

    def sample_buffer_uniform(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)

        batch_size = min(batch_size, max_mem)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        action_masks = self.action_mask_memory[batch]
        next_action_masks = self.next_action_mask_memory[batch]

        return states, actions, rewards, states_, dones, action_masks, next_action_masks
