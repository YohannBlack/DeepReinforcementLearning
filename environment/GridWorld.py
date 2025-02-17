import numpy as np
from typing import Tuple, Dict, Any
from environment.base import Environment


class GridWorld(Environment):
    def __init__(self, width: int = 5, height: int = 5,
                 initial_position: Tuple[int, int] = None,
                 goal_position: Tuple[int, int] = None):
        super().__init__()
        self.width = width
        self.height = height
        self.initial_position = initial_position or (0, 0)
        self.goal_position = goal_position or (width-1, height-1)
        self.current_position = None
        self.win_status = 0
        self.done = False

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.action_dict = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)
        }

    def reset(self) -> np.ndarray:
        self.current_position = self.initial_position
        self.episode_steps = 0
        self.total_reward = 0.0
        self.done = False
        return np.array(self.current_position)

    def _is_valid_position(self, y: int, x: int) -> bool:
        return 0 <= y < self.height and 0 <= x < self.width

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        info = {}
        self.episode_steps += 1

        dy, dx = self.action_dict[action]
        y, x = self.current_position

        next_y = max(0, min(y + dy, self.height - 1))
        next_x = max(0, min(x + dx, self.width - 1))

        if not self._is_valid_position(next_y, next_x):
            reward = -1
            self.done = False
            info = {'result': 'invalid'}
            self.win_status = -1
            return np.array(self.current_position), reward, self.done, info

        self.current_position = (next_y, next_x)

        reward = -0.1
        if self.current_position == self.goal_position:
            reward = 1
            self.done = True
            self.win_status = 1
            info = {'result': 'win'}
        elif self.episode_steps >= self.width * self.height * 2:  # Timeout
            self.done = True
            self.win_status = -1
            info = {'result': 'lose'}
        else:
            self.done = False

        self.total_reward += reward

        return np.array(self.current_position), reward, self.done, info

    def _get_valid_actions(self) -> np.ndarray:
        y, x = self.current_position
        valid_actions = []
        if y > 0:              # Can move up
            valid_actions.append(0)
        if x < self.width - 1:  # Can move right
            valid_actions.append(1)
        if y < self.height - 1:  # Can move down
            valid_actions.append(2)
        if x > 0:              # Can move left
            valid_actions.append(3)
        return valid_actions

    def is_done(self) -> bool:
        return self.done

    @property
    def action_space(self) -> Dict:
        return {
            'n': 4,  # Number of possible actions
            'values': [0, 1, 2, 3]  # Possible action values
        }

    @property
    def observation_space(self) -> Dict:
        return {
            'shape': (2,),  # 2D position
            'low': np.array([0, 0]),
            'high': np.array([self.height-1, self.width-1])
        }

    def render(self, mode: str = 'human') -> None:
        if mode == 'human':
            grid = [['-' for _ in range(self.width)]
                    for _ in range(self.height)]
            grid[self.current_position[0]][self.current_position[1]] = 'A'
            grid[self.goal_position[0]][self.goal_position[1]] = 'G'

            for row in grid:
                print(' '.join(row))
            print()


def play_gridworld():
    env = GridWorld(width=5, height=5)
    done = False
    obs = env.reset()

    while not done:
        env.render()
        print("Choose action from 0-3 (0: up, 1: right, 2: down, 3: left): ")
        action = int(input())
        obs, reward, done, info = env.step(action)
        if done:
            env.render()
