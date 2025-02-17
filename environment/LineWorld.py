import numpy as np
from typing import Tuple, Dict, Any
from environment.base import Environment


class LineWorld(Environment):
    def __init__(self, size: int = 10, initial_position: int = 0, goal_position: int = None):
        super().__init__()
        self.size = size
        self.initial_position = initial_position
        self.goal_position = goal_position if goal_position is not None else size - 1
        self.current_position = None
        self.win_status = 0
        self.done = False

        self.action_dict = {0: -1, 1: 1}

    def reset(self) -> np.ndarray:
        self.current_position = self.initial_position
        self.episode_steps = 0
        self.total_reward = 0.0
        self.done = False
        return np.array([self.current_position])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        info = {}
        self.episode_steps += 1

        movement = self.action_dict[action]
        next_position = self.current_position + movement

        self.current_position = np.clip(next_position, 0, self.size - 1)

        reward = -.1
        if self.current_position == self.goal_position:
            reward = 1.0
            self.done = True
            info = {'result': 'win'}
        elif self.episode_steps >= self.size * 2:
            self.done = True
            info = {'result': 'lose'}
        else:
            self.done = False

        self.total_reward += reward

        return np.array([self.current_position]), reward, self.done, info

    def _get_valid_actions(self) -> np.ndarray:
        valid_actions = []

        if self.current_position > 0:
            valid_actions.append(0)
        if self.current_position < self.size - 1:
            valid_actions.append(1)

        return valid_actions

    def is_done(self) -> bool:
        return self.done

    @property
    def action_space(self) -> Dict:
        return {
            'n': 2,
            'values': [0, 1]
        }

    @property
    def observation_space(self) -> Dict:
        return {
            'shape': (1,),
            'low': 0,
            'high': self.size - 1
        }

    def render(self, mode: str = 'human') -> None:
        if mode == 'human':
            line = ['-'] * self.size
            line[self.current_position] = 'A'
            line[self.goal_position] = 'G'
            print(''.join(line))


def play_lineworld():
    env = LineWorld(size=10)
    obs = env.reset()
    env.render()

    while not env.is_done():
        env.render()
        print("Select action: 0 (left) or 1 (right)")
        action = int(input())
        obs, reward, done, info = env.step(action)
    env.render()
