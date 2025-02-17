import numpy as np
from typing import Tuple, Dict, Any

from .base import Environment


class TicTacToe(Environment):
    def __init__(self):
        self.board = None
        self.player_symbol = 1
        self.opponent_symbol = -1
        self.done = False

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.episode_steps = 0
        self.total_reward = 0.0
        self.done = False
        return self.board.copy()

    def _is_winner(self, player: int) -> bool:
        board = self.board
        return (any(np.all(board[i, :] == player) for i in range(3)) or  # Rows
                any(np.all(board[:, i] == player) for i in range(3)) or  # Cols
                np.all(np.diag(board) == player) or  # Main diagonal
                np.all(np.diag(np.fliplr(board)) == player))  # Anti-diagonal

    def _get_valid_moves(self) -> list:
        return list(zip(*np.where(self.board == 0)))

    def _get_valid_actions(self) -> np.ndarray:
        empty_cells = np.where(self.board == 0)
        valid_actions = []

        for row, col in zip(*empty_cells):
            action_index = row * 3 + col
            valid_actions.append(action_index)

        return np.array(valid_actions)

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool]:
        self.episode_steps += 1
        row, col = action

        if self.board[row, col] != 0:
            self.done = True
            return self.board.copy(), -1., self.done, {'result': 'invalid'}

        self.board[row, col] = self.player_symbol

        if self._is_winner(self.player_symbol):
            self.done = True
            return self.board.copy(), 1., self.done, {'result': 'win'}

        valid_moves = self._get_valid_moves()
        if not valid_moves:
            self.done = True
            return self.board.copy(), 0.5, self.done, {'result': 'draw'}

        opp_row, opp_col = valid_moves[np.random.choice(len(valid_moves))]
        self.board[opp_row, opp_col] = self.opponent_symbol

        if self._is_winner(self.opponent_symbol):
            self.done = True
            return self.board.copy(), -1., self.done, {'result': 'lose'}

        self.done = False
        return self.board.copy(), -0.1, self.done, {}

    def is_done(self) -> bool:
        return self.done

    @property
    def action_space(self):
        return {
            'n': 9,
            'type': 'tuple',
            'values': [(i, j) for i in range(3) for j in range(3)]
        }

    @property
    def observation_space(self):
        return {
            'shape': (3, 3),
            'low': -1,
            'high': 1
        }

    def render(self, mode: str = 'human') -> None:
        if mode == 'human':
            symbol_map = {0: '-', 1: 'X', -1: 'O'}
            for row in self.board:
                print(' '.join(symbol_map[cell] for cell in row))


def play_tictactoe_vs_random():
    env = TicTacToe()
    done = False
    obs = env.reset()

    while not done:
        env.render()
        print("Player's Turn")
        action = input("Enter row and column (0-2) separated by space: ")
        row, col = map(int, action.split())
        obs, reward, done, info = env.step((row, col))
        if done:
            env.render()
