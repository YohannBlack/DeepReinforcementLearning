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
import time


class PolicyNetwork(nn.Module):
    def __init__(self, observation_space, n_actions):
        super(PolicyNetwork, self).__init__()

        self.input_dims = observation_space['shape']
        input_size = np.prod(self.input_dims)

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, state, action_mask=None):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        # Apply action masking
        if action_mask is not None:
            # Set logits of invalid actions to large negative value
            invalid_action_mask = ~action_mask
            logits = logits.masked_fill(invalid_action_mask, float('-inf'))
        return F.softmax(logits, dim=-1)


class ReinforceAgent:
    def __init__(
            self,
            observation_space,
            action_space,
            learning_rate=1e-4,
            gamma=0.99,
    ):
        self.input_dims = observation_space['shape']
        self.n_actions = action_space['n']
        self.action_space = action_space

        self.gamma = gamma

        self.states = []
        self.actions = []
        self.rewards = []
        self.action_masks = []  # Store action masks for training

        self.policy_net = PolicyNetwork(observation_space, self.n_actions)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.policy_net.to(self.device)

    def store_transition(self, state, action, reward, action_mask):
        self.states.append(state)
        if isinstance(action, tuple):
            action = self.action_space['values'].index(action)
        self.actions.append(action)
        self.rewards.append(reward)
        self.action_masks.append(action_mask)

    def choose_action(self, state, action_mask, eval_mode=False):
        if len(state.shape) > 1:
            state = state.flatten()

        state = T.tensor(state, dtype=T.float).to(self.device)
        action_mask = T.tensor(action_mask, dtype=T.bool).to(self.device)
        # print(state)
        # print(action_mask)

        with T.no_grad():
            probs = self.policy_net(state, action_mask)
            # print(probs)
            if eval_mode:
                action = T.argmax(probs).item()
            else:
                action = T.multinomial(probs, 1).item()

        return self.action_space['values'][action]

    def learn(self):
        if len(self.states) == 0:
            return

        states = np.array([state.flatten() if len(state.shape)
                          > 1 else state for state in self.states])
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        action_masks = np.array(self.action_masks)

        returns = np.zeros_like(rewards)
        G = 0.

        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.gamma * G
            returns[i] = G

        states = T.tensor(states, dtype=T.float32).to(self.device)
        actions = T.tensor(actions).to(self.device)
        returns = T.tensor(returns, dtype=T.float32).to(self.device)
        action_masks = T.tensor(action_masks, dtype=T.bool).to(self.device)

        probs = self.policy_net(states, action_masks)
        selected_probs = probs[range(len(probs)), actions]

        loss = -T.mean(T.log(selected_probs) * returns)

        self.optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.)
        self.optimizer.step()

        self.states = []
        self.actions = []
        self.rewards = []
        self.action_masks = []

        return loss.item()

    def evaluate(self, env, num_episode: int = 100):
        eval_rewards, eval_steps = [], []
        wins, losses, draws = 0, 0, 0

        for _ in range(num_episode):
            state = env.reset()
            done = False
            total_rewards, steps = 0, 0

            while not done:
                valid_actions = env.get_valid_actions()
                action = self.choose_action(
                    state, valid_actions, eval_mode=True)
                next_state, reward, done, info = env.step(action)

                total_rewards += reward
                steps += 1
                state = next_state

            eval_rewards.append(total_rewards)
            eval_steps.append(steps)

            if info['result'] == 'win':
                wins += 1
            elif info['result'] in ['lose', 'invalid']:
                losses += 1
            else:
                draws += 1

        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_step': np.mean(eval_steps),
            'win_rate': wins / num_episode,
            'loss_rate': losses / num_episode,
            'draw_rate': draws / num_episode
        }

    def train(self, env, episodes: int = 10000, eval_frequency: int = 1000,
              num_eval_episodes: int = 10000, save_dir: str = 'results/reinforce'):
        os.makedirs(save_dir, exist_ok=True)

        training_rewards, eval_metrics, eval_episodes, losses = [], [], [], []

        for episode in tqdm(range(episodes)):
            state = env.reset()
            episode_reward = 0.
            episode_loss = 0.
            done = False

            while not done:
                valid_actions = env.get_valid_actions()  # Get valid actions mask
                action = self.choose_action(state, valid_actions)
                next_state, reward, done, _ = env.step(action)

                self.store_transition(state, action, reward, valid_actions)
                episode_reward += reward
                state = next_state

            loss = self.learn()
            if loss is not None:
                episode_loss = loss

            training_rewards.append(episode_reward)
            losses.append(episode_loss)

            if episode % eval_frequency == 0 and episode > 0:
                metrics = self.evaluate(env, num_eval_episodes)
                metrics['episode'] = episode
                eval_metrics.append(metrics)
                eval_episodes.append(episode)

                print(f"\nEpisode: {episode}")
                print(
                    f"Average Training Reward: {np.mean(training_rewards[-100:]):.4f}")
                print(f"Win rate: {metrics['win_rate']}")

        results = {
            'training_rewards': training_rewards,
            'eval_metrics': eval_metrics,
            'parameters': {
                'gamma': self.gamma,
                'learning_rate': self.optimizer.param_groups[0]['lr']
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
        model_path = os.path.join('models', *model_dir,
                                  f'model_{timestamp}.pth')

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        T.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, model_path)

    def _load_model(self, model_path):
        checkpoint = T.load(model_path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _plot_metrics(self, training_rewards, eval_metrics, policy_losses,
                      eval_episodes, save_dir, episode, timestamp):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))

        # Plot training rewards
        ax1.plot(training_rewards, alpha=0.6)
        ax1.plot(np.convolve(training_rewards, np.ones(100)/100, mode='valid'),
                 label='Moving Average')
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()

        # Plot losses
        ax2.plot(policy_losses, alpha=0.6, label='Policy Loss')
        ax2.plot(np.convolve(policy_losses, np.ones(100)/100, mode='valid'),
                 label='Policy Loss MA')
        ax2.set_title('Losses')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()

        # Plot evaluation metrics
        ax3_twin = ax3.twinx()

        mean_rewards = [m['mean_reward'] for m in eval_metrics]
        std_rewards = [m['std_reward'] for m in eval_metrics]
        mean_steps = [m['mean_step'] for m in eval_metrics]
        reward_line = ax3.errorbar(eval_episodes, mean_rewards, yerr=std_rewards,
                                   label='Eval Reward', capsize=5, color='blue')
        step_line = ax3_twin.plot(eval_episodes, mean_steps, label='Mean Steps',
                                  color='red')
        ax3.set_title('Evaluation Metrics')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Mean Reward', color='blue')

        lines = [reward_line[0]] + step_line
        labels = [l.get_label() for l in lines]
        labels[0] = 'Mean Reward'
        ax3.legend()

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


def play_farkle_reinforce(env, model_path):
    agent = ReinforceAgent(env.observation_space, env.action_space)
    agent._load_model(model_path)

    state = env.reset()
    done = False
    total_rewards = 0

    while not done:
        env.render()
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(state, valid_actions, eval_mode=True)
        print(f"Selected action: {env.action_map[action]}")
        next_state, reward, done, info = env.step(action)

        state = next_state
        total_rewards += reward
        time.sleep(1)

    print(f"Game result: {info['result']}")
    print(f"Agent score: {env.agent.total_score}")
