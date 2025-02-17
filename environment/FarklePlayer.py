from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Player:
    total_score: int = 0
    current_score: int = 0
    selected_dice: List[int] = None
    hot_dice: bool = False
    consecutive_rolls: int = 0
    max_rolls: int = 1

    def __post_init__(self):
        if self.selected_dice is None:
            self.selected_dice = []

    def reset(self):
        self.total_score = 0
        self.current_score = 0
        self.selected_dice = []
        self.hot_dice = False
        self.consecutive_rolls = 0

    def bank_points(self) -> int:
        self.total_score += self.current_score
        banked_points = self.current_score
        self.current_score = 0
        self.selected_dice = []
        self.hot_dice = False
        self.consecutive_rolls = 0
        return banked_points

    def lose_turn(self):
        self.current_score = 0
        self.selected_dice = []
        self.hot_dice = False
        self.consecutive_rolls = 0

    def add_selected_dice(self, dice: np.ndarray, score: int):
        self.selected_dice.extend(dice)
        self.current_score += score
        self.consecutive_rolls = 0


class RandomPlayer(Player):
    def choose_action(self, available_dice: np.ndarray, remaining_dice: int) -> tuple:

        if self.current_score >= 1000:
            bank_threshold = 0.7
        elif self.current_score >= 500:
            bank_threshold = 0.5
        else:
            bank_threshold = 0.3

        if not self.hot_dice and self.current_score > 0 and np.random.random() < bank_threshold:
            return 'bank', None

        if remaining_dice == 0:
            return 'roll', None

        if len(available_dice) > 0:
            if np.random.random() < 0.7:
                return 'keep', np.random.choice(available_dice)

        return 'roll', None
