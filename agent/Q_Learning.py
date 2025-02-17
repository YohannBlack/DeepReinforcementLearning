import numpy as np
from typing import Tuple, Dict, Any
import random
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json
import os


class QLearningAgent:
    def __init__(self, action_space: Dict, observation_space: Dict,
                 learning_rate: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        self.action_space = action_space
        self.observation_space = observation_space
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.epsilon_evolution = [self.epsilon]

    def _get_state_key(self, state: np.ndarray) -> str:
        return str(state.tolist())

    def _ensure_state_exists(self, state: np.ndarray):
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {str(action): 0.0
                                       for action in self.action_space['values']}

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> Any:
        self._ensure_state_exists(state)

        if not eval_mode and random.random() < self.epsilon:
            return random.choice(self.action_space['values'])

        state_key = self._get_state_key(state)
        return eval(max(self.q_table[state_key].items(),
                        key=lambda x: x[1])[0])

    def update(self, state: np.ndarray, action: Any, reward: float, next_state: np.ndarray):
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)

        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        next_max_q = max(self.q_table[next_state_key].values())
        current_q = self.q_table[state_key][str(action)]
        new_q = current_q + self.lr * \
            (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][str(action)] = new_q

    def evaluate(self, env, num_eval_episode: int = 100) -> Dict:
        eval_rewards, eval_steps = [], []
        wins, losses, draws = 0, 0, 0

        for episode in range(num_eval_episode):
            state = env.reset()
            done = False
            steps = 0
            episode_reward = 0

            while not done:
                action = self.select_action(state, eval_mode=True)
                next_state, reward, done, info = env.step(action)

                episode_reward += reward
                steps += 1
                state = next_state

            eval_rewards.append(episode_reward)
            eval_steps.append(steps)

            if info['result'] == 'win':
                wins += 1
            elif info['result'] in ['lose', 'invalid']:
                losses += 1
            else:
                draws += 1

        return {
            'win_rate': wins / num_eval_episode,
            'loss_rate': losses / num_eval_episode,
            'draw_rate': draws / num_eval_episode,
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_steps': np.mean(eval_steps),
        }

    def train(self, env, episodes: int = 10000, eval_frequency: int = 1000,
              num_eval_episode: int = 100, save_dir: str = 'results'):
        os.makedirs(save_dir, exist_ok=True)

        training_rewards = []
        eval_metrics = []
        eval_episodes = []

        for episode in tqdm(range(episodes)):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state)

                state = next_state
                episode_reward += reward

            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.epsilon_evolution.append(self.epsilon)
            training_rewards.append(episode_reward)

            if episode % eval_frequency == 0 and episode > 0:
                eval_metric = self.evaluate(env, num_eval_episode)
                eval_metric['episode'] = episode
                eval_metrics.append(eval_metric)
                eval_episodes.append(episode)

                print(f"\nEpisode: {episode}")
                print(
                    f"Average Training reward: {np.mean(training_rewards[-eval_frequency:]):.2f}")
                print(f"Win rate: {eval_metric['win_rate']:.2f}")
                print(f"Epsilon: {self.epsilon:.2f}")

        results = {
            'training_rewards': training_rewards,
            'eval_metrics': eval_metrics,
            'parameters': {
                'learning_rate': self.lr,
                'gamma': self.gamma,
                'initial_epsilon': 1.0,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            }
        }

        datetime_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._save_results(results, save_dir, timestamp=datetime_str)
        self._plot_metrics(training_rewards, eval_metrics, eval_episodes,
                           save_dir, episode, timestamp=datetime_str)

        return results

    def _save_results(self, results, save_dir: str, timestamp: str):
        results_path = os.path.join(save_dir, f'results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)

        model_dir = save_dir.split('/')[1:]
        model_path = os.path.join('models', *model_dir,
                                  f'model_{timestamp}.json')

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, 'w') as f:
            json.dump(self.q_table, f, indent=4)

    def _load_model(self, model_path: str):
        with open(model_path, 'r') as f:
            self.q_table = json.load(f)

    def _plot_metrics(self, training_rewards: list, eval_metrics: list,
                      eval_episodes: list, save_dir: str, episode: int, timestamp: str):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

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
        plt.savefig(
            f'{save_dir}/training_plots_{(episode+1)//1000}k_{timestamp}.png')
        plt.close()


def play_QLearning(env, model_path):
    agent = QLearningAgent(env.action_space, env.observation_space)
    agent._load_model(model_path)

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        state = next_state

    env.render()
    print(f"Game result: {info['result']}")
    print(f"Total reward: {total_reward}")
