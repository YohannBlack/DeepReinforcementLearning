import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from copy import deepcopy


class MCTSNode:
    def __init__(self, state, done=False):
        self.state = state
        self.done = done
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior_probality = defaultdict(float)

    @property
    def value(self):
        return self.value_sum / (self.visits + 1e-8)

    def select_action(self, c=1.0):
        best_score = float('-inf')
        best_action = None

        for action in self.children:
            child = self.children[action]

            score = child.value + c * \
                child.prior_probality[action] * \
                np.sqrt(self.visits) / (1 + child.visits)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action


class ExItNetwork(nn.Module):
    def __init__(self, observation_space, n_actions):
        super(ExItNetwork, self).__init__()

        self.input_dims = observation_space['shape']
        input_size = np.prod(self.input_dims)

        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, state):
        x = state
        shared = self.shared(x)
        policy = self.policy(shared)
        value = self.value(shared)
        return policy, value


class ExpertApprenticeAgent:
    def __init__(
            self,
            env,
            observation_space,
            action_space,
            learning_rate=1e-4,
            gamma=0.99,
            n_simulations=100,
            c=1.0,
            temperature=1.0
    ):

        self.env = env
        self.input_dims = observation_space['shape']
        self.n_actions = action_space['n']
        self.action_space = action_space
        self.action_values = action_space['values']

        self.gamma = gamma
        self.n_simulations = n_simulations
        self.c = c
        self.temperature = temperature

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.network = ExItNetwork(
            observation_space, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=learning_rate)

    def mcts_search(self, root_state):
        root = MCTSNode(root_state)

        if len(root_state.shape) > 1:
            root_state = root_state.flatten()
        root_state = T.tensor(root_state, dtype=T.float32).to(self.device)

        with T.no_grad():
            policy, value = self.network(root_state)
            policy = policy.cpu().numpy()

        for i, action_val in enumerate(self.action_values):
            root.prior_probality[action_val] = policy[i]

        for _ in range(self.n_simulations):
            node = root
            search_path = [node]

            while node.children and not node.done:
                action = node.select_action(self.c)
                node = node.children[action]
                search_path.append(node)

            if not node.done:
                sim_env = deepcopy(self.env)

                if hasattr(sim_env, 'board'):  # TicTacToe
                    sim_env.board = node.state
                elif hasattr(sim_env, 'current_position'):
                    if len(node.state.shape) == 1:
                        if node.state.size == 1:
                            sim_env.current_position = int(node.state[0])
                        else:
                            sim_env.current_position = tuple(
                                map(int, node.state))
                    else:
                        raise ValueError(
                            f'Invalid state shape : {node.state.shape}')

                for action in self.action_values:
                    next_state, reward, done, _ = sim_env.step(action)
                    child = MCTSNode(next_state, done)
                    node.children[action] = child

                    if len(next_state.shape) > 1:
                        next_state = next_state.flatten()
                    next_state = T.tensor(
                        next_state, dtype=T.float32).to(self.device)
                    with T.no_grad():
                        policy, child_value = self.network(next_state)
                        policy = policy.cpu().numpy()

                    for i, action_val in enumerate(self.action_values):
                        child.prior_probality[action_val] = policy[i]

            value = 0 if node.done else child_value.item()
            for node in reversed(search_path):
                node.visits += 1
                node.value_sum += value
                value = reward + self.gamma * value
        return root

    def choose_action(self, state, eval_mode=False):
        root = self.mcts_search(state)

        visits = np.array([root.children[a].visits if a in root.children else 0
                           for a in self.action_values])

        if np.sum(visits) == 0:
            if len(state.shape) > 1:
                state = state.flatten()
            state = T.tensor(state, dtype=T.float32).to(self.device)
            with T.no_grad():
                policy, _ = self.network(state)
            probs = policy.cpu().numpy()
        else:
            visits = visits ** (1 / self.temperature)
            probs = visits / np.sum(visits + 1)

        probs = np.clip(probs, 0, 1)
        probs /= np.sum(probs)

        if eval_mode:
            return self.action_values[np.argmax(probs)]
        else:
            choice = np.random.choice(self.n_actions, p=probs)
            return self.action_values[choice]

    def learn(self, state, action, reward, next_state, done):
        if len(state.shape) > 1:
            state = state.flatten()
            next_state = next_state.flatten()

        state = T.tensor(state, dtype=T.float32).to(self.device)
        next_state = T.tensor(next_state, dtype=T.float32).to(self.device)

        root = self.mcts_search(state)
        visits = np.array([root.children[a].visits if a in root.children else 0
                           for a in self.action_values])
        improved_policy = visits / np.sum(visits)

        policy, value = self.network(state)
        with T.no_grad():
            _, next_value = self.network(next_state)

        target_value = reward + self.gamma * next_value * (1 - done)

        policy_loss = -T.sum(T.tensor(improved_policy, dtype=T.float32).to(self.device) *
                             T.log(policy + 1e-8))
        value_loss = F.mse_loss(value, target_value)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def train(self, episodes=10000, eval_frequency=1000, num_eval_episodes=100):
        training_rewards = []
        eval_metrics = []

        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                loss = self.learn(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

            training_rewards.append(total_reward)

            if episode % eval_frequency == 0:
                print(f"Episode {episode}")
                print(
                    f"Average reward: {np.mean(training_rewards[-eval_frequency:])}")

        return training_rewards, eval_metrics
