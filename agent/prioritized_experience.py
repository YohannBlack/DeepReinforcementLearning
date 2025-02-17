import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Stores priorities
        self.data_pointer = 0

    def propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):  # If we reach bottom, end the search
            return idx

        if s <= self.tree[left]:  # If left child has enough priority sum
            return self.retrieve(left, s)
        else:  # Otherwise, go right and subtract left sum
            return self.retrieve(right, s - self.tree[left])

    def add(self, priority, data_idx):
        tree_idx = data_idx + self.capacity - 1  # Index in tree array
        self.update(tree_idx, priority)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self.propagate(tree_idx, change)

    def get_leaf(self, v):
        idx = self.retrieve(0, v)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], data_idx

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.max_size = max_size
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.mem_counter = 0

        # PER hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6

        self.tree = SumTree(max_size)

        self.state_memory = np.zeros(
            (max_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (max_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(max_size, dtype=np.int64)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=np.uint8)

        self.max_priority = 1.0

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_counter % self.max_size

        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        priority = self.max_priority ** self.alpha
        self.tree.add(priority, idx)

        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        states = np.zeros((batch_size, *self.input_shape))
        actions = np.zeros(batch_size, dtype=np.int64)
        rewards = np.zeros(batch_size)
        states_ = np.zeros((batch_size, *self.input_shape))
        dones = np.zeros(batch_size, dtype=np.uint8)

        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size)

        segment = self.tree.total_priority / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            cumsum = np.random.uniform(a, b)

            tree_idx, priority, data_idx = self.tree.get_leaf(cumsum)

            indices[i] = data_idx
            priorities[i] = priority

            states[i] = self.state_memory[data_idx]
            actions[i] = self.action_memory[data_idx]
            rewards[i] = self.reward_memory[data_idx]
            states_[i] = self.new_state_memory[data_idx]
            dones[i] = self.terminal_memory[data_idx]

        sampling_probabilities = priorities / self.tree.total_priority
        is_weights = np.power(
            self.max_size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return (states, actions, rewards, states_, dones,
                indices, is_weights)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx + self.max_size - 1, priority)
            self.max_priority = max(self.max_priority, priority)
