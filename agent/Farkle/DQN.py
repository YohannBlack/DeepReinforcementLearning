import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
import json

from .replay_buffer import ReplayBuffer

has_gpu = T.cuda.is_available()
device = "cuda" if has_gpu else "cpu"


class DQNetwork(nn.Module):
    def __init__(self, observation_space, n_actions, batch_size):
        super(DQNetwork, self).__init__()

        self.batch_size = batch_size
        self.input_dims = observation_space['shape']
        input_size = np.prod(self.input_dims)

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, state, action_mask=None):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        if action_mask is not None:
            action_mask = T.tensor(action_mask, dtype=bool).to(device)
            actions = actions.masked_fill(~action_mask, float('-inf'))
        return actions


class DQNAgent:
    def __init__(
            self,
            observation_space,
            action_space,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.9995,
            memory_size=100000,
            batch_size=64,
            target_update=10,
    ):
        self.input_dims = observation_space['shape']
        self.n_actions = action_space['n']
        self.action_space = action_space

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon_evolution = []

        self.memory = ReplayBuffer(
            memory_size, self.input_dims, self.n_actions)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.policy_net = DQNetwork(
            observation_space, self.n_actions, batch_size).to(self.device)
        self.target_net = DQNetwork(
            observation_space, self.n_actions, batch_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate)

    def store_transition(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        self.memory.store_transition(
            state, action, reward, next_state, done, action_mask, next_action_mask)

    def choose_action(self, state, action_mask=None, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            # During exploration, only choose from valid actions if mask is provided
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                return self.action_space['values'][np.random.choice(valid_actions)]
            return np.random.choice(self.action_space['values'])

        if len(state.shape) > 1:
            state = state.flatten()

        state = T.tensor(state, dtype=T.float32).to(self.device)
        with T.no_grad():
            q_values = self.policy_net(state, action_mask)
        return self.action_space['values'][q_values.argmax().item()]

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        states, actions, rewards, next_states, dones, action_masks, next_action_masks = \
            self.memory.sample_buffer_uniform(self.batch_size)

        states = T.tensor(states, dtype=T.float32).to(self.device)
        states = states.reshape(self.batch_size, -1)
        actions = T.tensor(actions).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.device)
        next_states = T.tensor(next_states, dtype=T.float32).to(self.device)
        next_states = next_states.reshape(self.batch_size, -1)
        dones = T.tensor(dones, dtype=bool).to(self.device)
        action_masks = T.tensor(action_masks, dtype=bool).to(self.device)
        next_action_masks = T.tensor(
            next_action_masks, dtype=bool).to(self.device)

        current_q_values = self.policy_net(
            states, action_masks).gather(1, actions.unsqueeze(1))

        with T.no_grad():
            next_q_values = self.target_net(next_states, next_action_masks)
            next_q_values = next_q_values.max(1)[0]
            next_q_values[dones] = 0.0
            target_q_values = rewards + self.gamma * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def evaluate(self, env, num_episodes: int = 100):
        eval_rewards, eval_steps = [], []
        wins, losses, draws = 0, 0, 0

        for _ in range(num_episodes):
            state = env.reset()
            done = False
            total_reward, steps = 0, 0

            while not done:
                action_mask = env.get_valid_actions() if hasattr(
                    env, 'get_valid_actions') else None
                action = self.choose_action(state, action_mask, eval_mode=True)
                next_state, reward, done, info = env.step(action)

                total_reward += reward
                steps += 1
                state = next_state

            eval_rewards.append(total_reward)
            eval_steps.append(steps)

            if info['result'] == 'win':
                wins += 1
            elif info['result'] in ['lose', 'invalid']:
                losses += 1
            else:
                draws += 1

        return {
            'mean_reward': np.mean(eval_rewards),
            'mean_steps': np.mean(eval_steps),
            'std_reward': np.std(eval_rewards),
            'win_rate': wins / num_episodes,
            'loss_rate': losses / num_episodes,
            'draw_rate': draws / num_episodes
        }

    def train(self, env, episodes: int = 10000, eval_frequency: int = 1000,
              num_eval_episodes: int = 100, save_dir='results'):
        os.makedirs(save_dir, exist_ok=True)
        training_rewards, eval_metrics, eval_episodes, losses = [], [], [], []

        for episode in tqdm(range(episodes)):
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            step_count = 0
            done = False

            while not done:
                action_mask = env.get_valid_actions() if hasattr(
                    env, 'get_valid_actions') else None
                action = self.choose_action(state, action_mask)
                next_state, reward, done, _ = env.step(action)

                next_action_mask = env.get_valid_actions() if hasattr(
                    env, 'get_valid_actions') else None

                if isinstance(action, tuple):
                    action = self.action_space['values'].index(action)

                self.store_transition(
                    state, action, reward, next_state, done, action_mask, next_action_mask)
                loss = self.learn()

                if loss is not None:
                    episode_loss += loss

                if step_count % self.target_update == 0:
                    self.update_target_network()

                state = next_state
                episode_reward += reward
                step_count += 1

            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.epsilon_evolution.append(self.epsilon)

            training_rewards.append(episode_reward)
            losses.append(episode_loss / step_count if step_count > 0 else 0)

            if episode % eval_frequency == 0 and episode > 0:
                metrics = self.evaluate(env, num_eval_episodes)
                metrics['episode'] = episode
                eval_metrics.append(metrics)
                eval_episodes.append(episode)

                print(f"\nEpisode: {episode}")
                print(
                    f"Average Training reward: {np.mean(training_rewards[-100:]):.2f}")
                print(f"Win rate: {metrics['win_rate']:.2f}")
                print(f"Epsilon: {self.epsilon:.2f}")

        results = {
            'training_rewards': training_rewards,
            'eval_metrics': eval_metrics,
            'parameters': {
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gamma': self.gamma,
                'initial_epsilon': 1.0,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min,
                'batch_size': self.batch_size,
                'target_update': self.target_update
            }
        }

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._save_results(results, save_dir, timestamp)
        self._plot_metrics(training_rewards, eval_metrics, losses,
                           eval_episodes, save_dir, episodes, timestamp)

        return results

    def _save_results(self, results, save_dir, timestamp):
        results_path = os.path.join(save_dir, f'results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        model_dir = save_dir.split('/')[1:]
        model_path = os.path.join(
            'models', *model_dir, f'model_{timestamp}.pth')

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        T.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, model_path)

    def _plot_metrics(self, training_rewards, eval_metrics, losses,
                      eval_episodes, save_dir, episode, timestamp):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))

        ax1_twin = ax1.twinx()
        reward_line = ax1.plot(training_rewards, alpha=0.6)
        moving_avg_line = ax1.plot(np.convolve(training_rewards, np.ones(100)/100, mode='valid'),
                                   label='Moving Average')
        epsilon_line = ax1_twin.plot(self.epsilon_evolution, color='red',
                                     label='Epsilon', linestyle='dashed')

        lines = reward_line + moving_avg_line + epsilon_line
        labels = [l.get_label() for l in lines]
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward', color='blue', alpha=0.6)
        ax1_twin.set_ylabel('Epsilon', color='red')
        ax1.legend(lines, labels)

        ax2.plot(losses, alpha=0.6)
        ax2.plot(np.convolve(losses, np.ones(100)/100, mode='valid'),
                 label='Moving Average')
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()

        ax3_twin = ax3.twinx()
        mean_rewards = [m['mean_reward'] for m in eval_metrics]
        std_rewards = [m['std_reward'] for m in eval_metrics]
        mean_steps = [m['mean_steps'] for m in eval_metrics]
        reward_line = ax3.errorbar(eval_episodes, mean_rewards, yerr=std_rewards,
                                   label='Eval Reward', capsize=5, color='blue')
        step_line = ax3_twin.plot(
            eval_episodes, mean_steps, label='Mean Steps', color='red')

        lines = [reward_line[0]] + step_line
        labels = [l.get_label() for l in lines]
        labels[0] = 'Mean Reward'
        ax3.set_title('Evaluation Metrics')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Mean Reward', color='blue')
        ax3_twin.set_ylabel('Mean Steps', color='red')
        ax3.legend(lines, labels)

        # Plot win/loss/draw rates
        win_rates = [m['win_rate'] for m in eval_metrics]
        loss_rates = [m['loss_rate'] for m in eval_metrics]
        draw_rates = [m['draw_rate'] for m in eval_metrics]

        ax4.plot(eval_episodes, win_rates, label='Win Rate')
        ax4.plot(eval_episodes, loss_rates, label='Loss Rate')
        ax4.plot(eval_episodes, draw_rates, label='Draw Rate')
        ax4.set_title('Game Outcomes')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Rate')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(
            f'{save_dir}/training_plots_{(episode+1)//1000}k_{timestamp}.png')
        plt.close()
