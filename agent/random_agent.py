import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, state, eval_mode=False):
        if type(self.action_space['values'][0]) == tuple:
            action_index = [i for i in range(len(self.action_space['values']))]
            action = np.random.choice(action_index)
            return self.action_space['values'][action]
        return np.random.choice(self.action_space['values'])

    def evaluate(self, env, num_episodes: int = 100):
        eval_rewards, eval_steps = [], []
        wins, losses, draws = 0, 0, 0

        for _ in range(num_episodes):
            state = env.reset()
            done = False
            total_reward, steps = 0, 0

            while not done:
                action = self.choose_action(state, eval_mode=True)
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

        for episode in tqdm(range(episodes)):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                state = next_state
                episode_reward += reward

            training_rewards.append(episode_reward)

            if episode % eval_frequency == 0:
                metrics = self.evaluate(env, num_eval_episodes)
                metrics['episode'] = episode
                eval_metrics.append(metrics)
                eval_episodes.append(episode)

                print(f"\nEpisode: {episode}")
                print(
                    f"Average Training reward: {np.mean(training_rewards[-eval_frequency:]):.2f}")
                print(f"Win rate: {metrics['win_rate']:.2f}")

        results = {
            'training_rewards': training_rewards,
            'eval_metrics': eval_metrics,
            'parameters': {
                'type': 'random'
            }
        }

        self._save_results(results, save_dir)
        self._plot_metrics(training_rewards, eval_metrics,
                           eval_episodes, save_dir, episodes)

        return results

    def _save_results(self, results, save_dir):
        results_path = os.path.join(save_dir, f'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

    def _plot_metrics(self, training_rewards: list, eval_metrics: list,
                      eval_episodes: list, save_dir: str, episode: int):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        # Plot training rewards
        ax1.plot(training_rewards, alpha=0.6)
        ax1.plot(np.convolve(training_rewards, np.ones(100)/100, mode='valid'),
                 label='Moving Average')
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()

        # Plot evaluation metrics
        ax2_twin = ax2.twinx()

        mean_rewards = [m['mean_reward'] for m in eval_metrics]
        std_rewards = [m['std_reward'] for m in eval_metrics]
        mean_steps = [m['mean_steps'] for m in eval_metrics]

        reward_line = ax2.errorbar(eval_episodes, mean_rewards, yerr=std_rewards,
                                   label='Eval Reward', capsize=5, color='blue')
        step_line = ax2_twin.plot(
            eval_episodes, mean_steps, label='Mean Steps', color='red')

        ax2.set_title('Evaluation Metrics')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Mean Reward', color='blue')
        ax2_twin.set_ylabel('Mean Steps', color='red')

        lines = [reward_line[0]] + step_line
        labels = [l.get_label() for l in lines]
        labels[0] = 'Mean Reward'
        ax2.legend(lines, labels)

        # Plot win/loss/draw rates
        win_rates = [m['win_rate'] for m in eval_metrics]
        loss_rates = [m['loss_rate'] for m in eval_metrics]
        draw_rates = [m['draw_rate'] for m in eval_metrics]

        ax3.plot(eval_episodes, win_rates, label='Win Rate')
        ax3.plot(eval_episodes, loss_rates, label='Loss Rate')
        ax3.plot(eval_episodes, draw_rates, label='Draw Rate')
        ax3.set_title('Game Outcomes')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Rate')
        ax3.legend()

        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_plots_{(episode+1)//1000}k.png')
        plt.close()
