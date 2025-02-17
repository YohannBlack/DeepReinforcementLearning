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
        self.dropout = nn.Dropout(0.6)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, state, action_mask=None):
        x = state
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc3(x)

        # Apply action masking
        if action_mask is not None:
            # Set logits of invalid actions to large negative value
            invalid_action_mask = ~action_mask
            logits = logits.masked_fill(invalid_action_mask, float('-inf'))

        return F.softmax(logits, dim=-1) + 1e-8


class ValueNetwork(nn.Module):
    def __init__(self, observation_space):
        super(ValueNetwork, self).__init__()

        self.input_dims = observation_space['shape']
        input_size = np.prod(self.input_dims)

        self.fc1 = nn.Linear(input_size, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        return self.fc3(x)


class ActorCriticAgent:
    def __init__(
            self,
            observation_space,
            action_space,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            gamma=0.99
    ):
        self.input_dims = observation_space['shape']
        self.n_actions = action_space['n']
        self.action_space = action_space
        self.gamma = gamma

        self.actor = PolicyNetwork(observation_space, self.n_actions)
        self.critic = ValueNetwork(observation_space)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=learning_rate_critic)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.policy_losses = []
        self.value_losses = []
        self.I = 1.0

    def choose_action(self, state, action_mask, eval_mode=False):
        if len(state.shape) > 1:
            state = state.flatten()

        state = T.tensor(state, dtype=T.float32).to(self.device)
        action_mask = T.tensor(action_mask, dtype=T.bool).to(self.device)

        with T.no_grad():
            action_probs = self.actor(state, action_mask)

        if eval_mode:
            action_idx = T.argmax(action_probs).item()
        else:
            action_idx = T.distributions.Categorical(
                action_probs).sample().item()

        return self.action_space['values'][action_idx]

    def learn(self, state, action, reward, next_state, done, action_mask):
        state = T.tensor(state, dtype=T.float32).to(self.device)
        if len(state.shape) > 1:
            state = state.flatten()
        next_state = T.tensor(next_state, dtype=T.float32).to(self.device)
        if len(next_state.shape) > 1:
            next_state = next_state.flatten()
        if isinstance(action, tuple):
            action = self.action_space['values'].index(action)
        action = T.tensor(action).to(self.device)
        reward = T.tensor(reward, dtype=T.float32).to(self.device)
        action_mask = T.tensor(action_mask, dtype=T.bool).to(self.device)

        value = self.critic(state)

        # Get next state value (0 if terminal)
        next_value = T.tensor(0.0) if done else self.critic(next_state)

        # Calculate TD error
        delta = reward + self.gamma * next_value - value

        # Update critic
        critic_loss = delta.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor using masked probabilities
        action_probs = self.actor(state, action_mask)
        dist = T.distributions.Categorical(action_probs)
        log_prob = dist.log_prob(action)
        actor_loss = -self.I * delta.detach() * log_prob

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.I *= self.gamma

        return actor_loss.item(), critic_loss.item()

    def evaluate(self, env, num_episodes: int = 100):
        eval_rewards, eval_steps = [], []
        wins, losses, draws = 0, 0, 0

        for _ in range(num_episodes):
            state = env.reset()
            done = False
            total_reward, steps = 0, 0

            while not done:
                valid_actions = env.get_valid_actions()
                action = self.choose_action(
                    state, valid_actions, eval_mode=True)
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

        training_rewards = []
        eval_metrics = []
        eval_episodes = []
        episode_policy_losses = []
        episode_value_losses = []

        for episode in tqdm(range(episodes)):
            state = env.reset()
            done = False
            episode_reward = 0
            policy_losses = []
            value_losses = []
            self.I = 1.0  # Reset importance sampling weight

            while not done:
                valid_actions = env.get_valid_actions()  # Get valid actions mask
                action = self.choose_action(state, valid_actions)
                next_state, reward, done, _ = env.step(action)

                policy_loss, value_loss = self.learn(
                    state, action, reward, next_state, done, valid_actions)

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)

                state = next_state
                episode_reward += reward

            training_rewards.append(episode_reward)
            episode_policy_losses.append(np.mean(policy_losses))
            episode_value_losses.append(np.mean(value_losses))

            if episode % eval_frequency == 0:
                metrics = self.evaluate(env, num_eval_episodes)
                metrics['episode'] = episode
                eval_metrics.append(metrics)
                eval_episodes.append(episode)

                print(f"\nEpisode: {episode}")
                print(
                    f"Average Training reward: {np.mean(training_rewards[-100:]):.2f}")
                print(f"Win rate: {metrics['win_rate']:.2f}")

        results = {
            'training_rewards': training_rewards,
            'eval_metrics': eval_metrics,
            'parameters': {
                'actor_learning_rate': self.actor_optimizer.param_groups[0]['lr'],
                'critic_learning_rate': self.critic_optimizer.param_groups[0]['lr'],
                'gamma': self.gamma
            }
        }

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self._save_results(results, save_dir, timestamp=timestamp)
        self._plot_metrics(training_rewards, eval_metrics, episode_policy_losses,
                           episode_value_losses, eval_episodes, save_dir, episodes, timestamp)

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
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, model_path)

    def _load_model(self, model_path):
        checkpoint = T.load(model_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    def _plot_metrics(self, training_rewards: list, eval_metrics: list,
                      policy_losses: list, value_losses: list,
                      eval_episodes: list, save_dir: str, episode: int, timestamp: str):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20))

        # Plot training rewards
        ax1.plot(training_rewards, alpha=0.6)
        ax1.plot(np.convolve(training_rewards, np.ones(100)/100, mode='valid'),
                 label='Moving Average')
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()

        ax2_twin = ax2.twinx()
        # Plot losses
        actor_loss = ax2.plot(policy_losses, alpha=0.6, label='Actor Loss')
        actor_cumsum = ax2.plot(np.convolve(policy_losses, np.ones(100)/100, mode='valid'),
                                label='Moving Average')
        critic_loss = ax2_twin.plot(value_losses, alpha=0.6,
                                    label='Critic Loss', color='navajowhite')
        critic_cumsum = ax2_twin.plot(np.convolve(value_losses, np.ones(100)/100, mode='valid'),
                                      label='Moving Average', color='darkgoldenrod')
        ax2.set_title('Training Losses')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')

        lines = actor_loss + critic_loss + actor_cumsum + critic_cumsum
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels)

        # Plot evaluation metrics
        ax3_twin = ax3.twinx()

        mean_rewards = [m['mean_reward'] for m in eval_metrics]
        std_rewards = [m['std_reward'] for m in eval_metrics]
        mean_steps = [m['mean_steps'] for m in eval_metrics]
        reward_line = ax3.errorbar(eval_episodes, mean_rewards, yerr=std_rewards,
                                   label='Eval Reward', capsize=5, color='blue')
        step_line = ax3_twin.plot(eval_episodes, mean_steps,
                                  label='Mean Steps', color='red')

        ax3.set_title('Evaluation Metrics')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Mean Reward', color='blue')
        ax3_twin.set_ylabel('Mean Steps', color='red')

        lines = [reward_line[0]] + step_line
        labels = [l.get_label() for l in lines]
        labels[0] = 'Eval Reward'
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


def play_farkle_ac(env, model_path):
    agent = ActorCriticAgent(env.observation_space, env.action_space)
    agent._load_model(model_path)

    state = env.reset()
    done = False

    while not done:
        env.render()
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(state, valid_actions, eval_mode=True)
        print(f"Agent chooses: {env.action_map[action]}")
        next_state, reward, done, info = env.step(action)

        state = next_state
        time.sleep(1)

    env.render()
    print(f"Final score: {env.agent.total_score}")
    print(f"Result: {info['result']}")
