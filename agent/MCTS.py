import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import random
import time
import os
import json
import matplotlib.pyplot as plt


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0
        self.is_terminal = False
        self.is_fully_expanded = self.is_terminal


class MCTSAgent:
    def __init__(self, env, search_time=2.0, exploration_constant=1.414):
        self.env = env
        self.search_time = search_time
        self.n_rollouts = 0
        self.exploration_constant = exploration_constant

    def search(self, state, need_detail=False):
        self.root = Node(state, None)

        start_time = time.process_time()
        while time.process_time() - start_time < self.search_time:
            self.execute_round()
            self.n_rollouts += 1

        best_child = self.get_best_child(self.root, 0)
        action = next(action for action,
                      node in self.root.children.items() if node is best_child)
        if need_detail:
            return {'action': action, 'value': best_child.total_reward / best_child.visits}
        return action

    def execute_round(self):
        node = self.select_node(self.root)
        reward = self.rollout(node.state)
        self.backpropagate(node, reward)

    def select_node(self, node):
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return self.expand(node)
            node = self.get_best_child(node, self.exploration_constant)
        return node

    def expand(self, node):
        env_copy = deepcopy(self.env)
        actions = self.env._get_valid_actions()

        for action in actions:
            if action not in node.children:
                # env_copy.reset()
                action = env_copy.action_space['values'][action]
                next_state, _, done, _ = env_copy.step(action)
                new_node = Node(next_state, node)
                new_node.is_terminal = done
                node.children[action] = new_node
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_node

    def rollout(self, state):
        env_copy = deepcopy(self.env)
        current_state = state.copy()
        done = False
        total_reward = 0

        while not done:
            action_index = env_copy._get_valid_actions()
            action = random.choice(action_index)
            action = env_copy.action_space['values'][action]
            next_state, reward, done, _ = env_copy.step(action)
            total_reward += reward
            current_state = next_state
        return total_reward

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def get_best_child(self, node, exploration_weight):
        best_score = float('-inf')
        best_children = []

        for action, child in node.children.items():
            if child.visits == 0:
                continue

            exploitation = child.total_reward / child.visits
            exploration = exploration_weight * \
                math.sqrt(math.log(node.visits) / child.visits)
            utc_value = exploitation + exploration

            if utc_value > best_score:
                best_children = [child]
                best_score = utc_value
            elif utc_value == best_score:
                best_children.append(child)

        return random.choice(best_children)


def play_game(env, agent, render=False):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    rollouts = []

    while not done:
        if render:
            env.render()

        details = agent.search(state, need_detail=True)
        action = details['action']
        rollouts.append(agent.n_rollouts)

        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    if render:
        env.render()

    return {
        'total_reward': total_reward,
        'steps': steps,
        'outcome': info.get('result', 'unknown'),
        'avg_n_rollouts': np.mean(rollouts),
        'final_state': state
    }


def play_multiple_games(env, n_games=100, render=False, save_dir='results/mcts/'):
    agent = MCTSAgent(env)

    rewards = []
    steps_list = []
    outcomes = {'win': 0, 'lose': 0, 'draw': 0, 'invalid': 0}
    rollouts_list = []

    for game in tqdm(range(n_games), desc="Playing games"):
        if game % 10 == 0:
            game_stats = play_game(env, agent, True)
        else:
            game_stats = play_game(env, agent)

        rewards.append(game_stats['total_reward'])
        steps_list.append(game_stats['steps'])
        outcomes[game_stats['outcome']] += 1
        rollouts_list.append(game_stats['avg_n_rollouts'])

    results = {
        'n_games': n_games,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_steps': np.mean(steps_list),
        'std_steps': np.std(steps_list),
        'mean_rollouts': np.mean(rollouts_list),
        'outcomes': outcomes,
        'win_rate': outcomes['win'] / n_games,
        'loss_rate': (outcomes['lose'] + outcomes['invalid']) / n_games,
        'draw_rate': outcomes['draw'] / n_games
    }
    save_results(results, save_dir)
    plot_mcts_results(results, save_dir)
    return results


def save_results(results: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'results.json')
    with open(filename, 'w') as f:
        json.dump(results, f)


def plot_mcts_results(results: dict, save_dir: str = 'results/mcts'):
    os.makedirs(save_dir, exist_ok=True)

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MCTS Evaluation Results', fontsize=16)

    # Plot 1: Game Outcomes
    outcomes = results['outcomes']
    ax1.bar(outcomes.keys(), outcomes.values())
    ax1.set_title('Game Outcomes')
    ax1.set_ylabel('Number of Games')
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Win/Loss/Draw Rates
    rates = ['win_rate', 'loss_rate', 'draw_rate']
    rate_values = [results[rate] for rate in rates]
    ax2.bar([r.replace('_rate', '') for r in rates], rate_values)
    ax2.set_title('Game Outcome Rates')
    ax2.set_ylabel('Rate')
    ax2.set_ylim(0, 1)

    # Plot 3: Steps Distribution
    ax3.boxplot([results['mean_steps']], labels=['Steps per Game'])
    ax3.set_title(
        f'Steps Distribution\nMean: {results["mean_steps"]:.2f} ± {results["std_steps"]:.2f}')

    # Plot 4: Reward Distribution
    ax4.boxplot([results['mean_reward']], labels=['Reward per Game'])
    ax4.set_title(
        f'Reward Distribution\nMean: {results["mean_reward"]:.2f} ± {results["std_reward"]:.2f}')

    # Add mean rollouts information as text
    fig.text(0.02, 0.02, f'Average Rollouts per Move: {results["mean_rollouts"]:.2f}',
             fontsize=10, transform=fig.transFigure)
    fig.text(0.02, 0.04, f'Total Games: {results["n_games"]}',
             fontsize=10, transform=fig.transFigure)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'mcts_plots.png'))
    plt.close()
