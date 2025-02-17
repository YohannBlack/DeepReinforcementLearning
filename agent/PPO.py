import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
import json

has_gpu = T.cuda.is_available()
device = "cuda" if has_gpu else "cpu"


class ActorCritic(nn.Module):
    def __init__(self, observation_space, n_actions):
        super(ActorCritic, self).__init__()
        input_size = np.prod(observation_space['shape'])

        self.actor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        action_probs = F.softmax(self.actor(state), dim=-1) + 1e-8

        state_value = self.critic(state)

        return action_probs, state_value


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches


class PPOAgent:
    def __init__(
            self,
            observation_space,
            action_space,
            learning_rate=0.0003,
            gamma=0.99,
            gae_lambda=0.95,
            policy_clip=0.2,
            epochs=4,
            batch_size=64
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_actions = action_space['n']
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.epochs = epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size

        self.policy = ActorCritic(observation_space, self.n_actions).to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': learning_rate},
            {'params': self.policy.critic.parameters(), 'lr': learning_rate}
        ])

        self.memory = PPOMemory(batch_size)

    def choose_action(self, state, eval_mode=False):
        if len(state.shape) > 1:
            state = state.flatten()

        state = T.tensor(state, dtype=T.float32).to(device)
        action_probs, value = self.policy(state)

        if eval_mode:
            action_idx = T.argmax(action_probs).item()
        else:
            dist = Categorical(action_probs)
            action_idx = dist.sample().item()

        return (self.action_space['values'][action_idx],
                action_probs[action_idx].item(),
                value.item())

    def learn(self):
        for _ in range(self.epochs):
            state_arr = T.tensor(np.array(self.memory.states),
                                 dtype=T.float32).to(device)
            action_arr = T.tensor(
                np.array(self.memory.actions), dtype=T.long).to(device)
            old_prob_arr = T.tensor(
                np.array(self.memory.probs), dtype=T.float32).to(device)
            vals_arr = T.tensor(np.array(self.memory.vals),
                                dtype=T.float32).to(device)

            returns = []
            advantages = []
            values = vals_arr.cpu().numpy()
            rewards = np.array(self.memory.rewards)
            dones = np.array(self.memory.dones)

            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.gamma * \
                    next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * \
                    self.gae_lambda * (1 - dones[t]) * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])

            advantages = T.tensor(advantages, dtype=T.float32).to(device)
            returns = T.tensor(returns, dtype=T.float32).to(device)

            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-8)

            for batch in self.memory.generate_batches():
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]
                returns_batch = returns[batch]
                advantages_batch = advantages[batch]

                action_probs, critic_value = self.policy(states)
                critic_value = critic_value.squeeze()

                dist = Categorical(action_probs)
                new_probs = dist.log_prob(actions).exp()

                prob_ratio = new_probs / old_probs

                weighted_probs = advantages_batch * prob_ratio
                weighted_clipped_probs = advantages_batch * T.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip)

                actor_loss = -T.min(weighted_probs,
                                    weighted_clipped_probs).mean()

                critic_loss = F.mse_loss(critic_value, returns_batch)

                total_loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                T.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.)
                self.optimizer.step()

        self.memory.clear_memory()
        return total_loss.item()

    def evaluate(self, env, num_episodes=100):
        eval_rewards, eval_steps = [], []
        wins, losses, draws = 0, 0, 0

        for _ in range(num_episodes):
            state = env.reset()
            done = False
            total_reward, steps = 0, 0

            while not done:
                action, _, _ = self.choose_action(state, eval_mode=True)
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

    def train(self, env, episodes=10000, eval_frequency=1000,
              num_eval_episodes=100, save_dir='results'):
        os.makedirs(save_dir, exist_ok=True)

        training_rewards, eval_metrics, eval_episodes, losses = [], [], [], []

        for episode in tqdm(range(episodes)):
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            step_count = 0
            done = False

            while not done:
                if len(state.shape) > 1:
                    state = state.flatten()

                action, prob, val = self.choose_action(state)

                if isinstance(action, tuple):
                    action_idx = self.action_space['values'].index(action)
                else:
                    action_idx = action

                next_state, reward, done, _ = env.step(action)

                self.memory.store_memory(state,
                                         action_idx,
                                         prob,
                                         val,
                                         reward,
                                         done)

                state = next_state
                episode_reward += reward
                step_count += 1

            if len(self.memory.states) > self.memory.batch_size:
                episode_loss = self.learn()

            training_rewards.append(episode_reward)
            losses.append(episode_loss)

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
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gamma': self.gamma,
                'policy_clip': self.policy_clip,
                'epochs': self.epochs,
                'batch_size': self.batch_size
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
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, model_path)

    def _load_model(self, model_path: str):
        checkpoint = T.load(model_path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _plot_metrics(self, training_rewards: list, eval_metrics: list,
                      policy_losses: list, eval_episodes: list, save_dir: str, episodes: int, timestamp: str):
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
        ax2.plot(policy_losses, alpha=0.6, color='navajowhite')
        ax2.plot(np.convolve(policy_losses, np.ones(100)/100, mode='valid'),
                 label='Moving Average', color='darkgoldenrod')
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend()

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
        ax3.set_ylabel('Mean Reward')

        lines = [reward_line[0]] + step_line
        labels = [l.get_label() for l in lines]
        labels[0] = 'Mean Reward'
        ax3.legend(lines, labels)

        # Plot win rate
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
            f'{save_dir}/training_plots_{(episodes+1)//1000}k_{timestamp}.png')
        plt.close()


def play_ppo(env, model_path):
    agent = PPOAgent(env.observation_space, env.action_space)
    agent._load_model(model_path)

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action, _, _ = agent.choose_action(state, eval_mode=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward

    env.render()
    print(f"Game result: {info['result']}")
    print(f"Total reward: {total_reward}")
