import numpy as np
from collections import Counter
from typing import Tuple, Dict, Any
from environment.base import Environment
from environment.FarklePlayer import Player, RandomPlayer
import time


class Farkle(Environment):
    def __init__(self):
        super(Farkle, self).__init__()
        self.n_dice = 6
        self.score_to_win = 5000
        self.dice = None
        self.remaining_dice = self.n_dice

        self.agent = Player()
        self.opponent = RandomPlayer()
        self.current_player = self.agent

        self.action_map = {
            0: 'roll',
            1: 'bank'
        }

        for i in range(1, 7):
            self.action_map[i + 1] = f'keep_{i}'

    def _roll_dice(self, n: int) -> np.ndarray:
        return np.random.randint(1, 7, n)

    def get_valid_actions(self) -> np.ndarray:
        valid_actions = np.zeros(8, dtype=bool)

        if self.agent.current_score > 0:
            valid_actions[1] = True

        if len(self.agent.selected_dice) > 0:
            valid_actions[0] = True

        if self.dice is not None:
            for value in range(1, 7):
                dice_indices = np.where(self.dice == value)[0]
                if len(dice_indices) == 0:
                    continue

                test_dice = self.dice[dice_indices]
                score = self._calculate_score(test_dice)
                if score > 0:
                    valid_actions[value + 1] = True

        if self.remaining_dice == 6 and self._calculate_score(self.dice) >= 500:
            valid_actions[1] = True

        return valid_actions

    def _calculate_score(self, dice: np.ndarray) -> int:
        counter = Counter(dice)
        score = 0

        # Straight
        if len(counter) == 6 and len(counter) == 6:
            return 1500

        # Three pairs
        if len(dice) == 6 and len(counter) == 3 and all(v == 2 for v in counter.values()):
            return 1000

        # No scoring combination
        if len(dice) == 6 and len(counter) > 0:
            if not any(count >= 3 for count in counter.values()) and 1 not in counter and 5 not in counter:
                return 500

        for value, count in counter.items():
            if count >= 3:
                base_score = 1000 if value == 1 else value * 100
                if count == 3:
                    score += base_score
                elif count == 4:
                    score += base_score * 2
                elif count == 5:
                    score += base_score * 4
                elif count == 6:
                    score += base_score * 8

                counter[value] -= (count - (count % 3))

        score += counter[1] * 100
        score += counter[5] * 50

        return score

    def _play_opponent_turn(self) -> int:
        while True:
            action_type, dice_value = self.opponent.choose_action(
                self.dice, self.remaining_dice)
            # print(action_type)

            if action_type == 'bank':
                self.opponent.bank_points()
                self.remaining_dice = self.n_dice
                self.dice = self._roll_dice(self.n_dice)
                self.current_player = self.agent
                return

            elif action_type == 'roll':
                self.dice = self._roll_dice(self.remaining_dice)
                potential_score = self._calculate_score(self.dice)

                if potential_score == 0:
                    self.opponent.lose_turn()
                    self.current_player = self.agent
                    self.remaining_dice = self.n_dice
                    self.dice = self._roll_dice(self.n_dice)
                    return
            elif action_type == 'keep':
                kept_dice = self.dice[self.dice == dice_value]
                new_score = self._calculate_score(kept_dice)

                if new_score > 0:
                    self.opponent.add_selected_dice(kept_dice, new_score)
                    self.remaining_dice -= len(kept_dice)
                    self.dice = self.dice[self.dice != dice_value]

                    if self.remaining_dice == 0:
                        if self._calculate_score(self.opponent.selected_dice) > 0:
                            self.opponent.hot_dice = True
                            self.remaining_dice = self.n_dice
                            self.opponent.selected_dice = []


    def reset(self) -> np.ndarray:
        self.agent.reset()
        self.opponent.reset()
        self.dice = self._roll_dice(self.n_dice)
        self.remaining_dice = self.n_dice
        self.current_player = self.agent
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action_type = self.action_map[action]
        reward = -0.1

        if self._calculate_score(self.dice) >= 500:
            reward = 0.1

        if action_type == 'roll':
            if len(self.agent.selected_dice) == 0 and not self.agent.hot_dice:
                return self.get_state(), -1, True, {'result': 'invalid'}

            self.dice = self._roll_dice(self.remaining_dice)
            potential_score = self._calculate_score(self.dice)

            if potential_score == 0:  # Farkle
                self.agent.lose_turn()
                self.current_player = self.opponent
                self.remaining_dice = self.n_dice
                self.dice = self._roll_dice(self.n_dice)
                self._play_opponent_turn()

                if self.opponent.total_score >= self.score_to_win:
                    return self.get_state(), -1, True, {'result': 'lose'}
                return self.get_state(), -0.1, False, {}

        elif action_type == 'bank':
            if self.remaining_dice == 6 and self._calculate_score(self.dice) >= 500:
                self.agent.current_score += 500
                self.agent.bank_points()
                self.remaining_dice = self.n_dice
                self._roll_dice(self.n_dice)
                self._play_opponent_turn()
                return self.get_state(), 0, False, {}

            if self.agent.current_score == 0:
                return self.get_state(), -1, True, {'result': 'invalid'}

            banked_points = self.agent.bank_points()
            reward = min(1.0, banked_points / self.score_to_win)

            if self.agent.total_score >= self.score_to_win:
                return self.get_state(), 1, True, {'result': 'win'}

            self.current_player = self.opponent
            self.remaining_dice = self.n_dice
            self.dice = self._roll_dice(self.n_dice)
            self._play_opponent_turn()

            if self.opponent.total_score >= self.score_to_win:
                return self.get_state(), -1, True, {'result': 'lose'}

            self.remaining_dice = self.n_dice
            self.dice = self._roll_dice(self.n_dice)
            return self.get_state(), reward, False, {}

        else:
            value = int(action_type.split('_')[1])
            dice_indices = np.where(self.dice == value)[0]
            if len(dice_indices) == 0:
                return self.get_state(), -1, True, {'result': 'invalid'}

            selected = self.dice[dice_indices]
            score = self._calculate_score(selected)

            if score == 0:
                return self.get_state(), -1, True, {'result': 'invalid'}

            self.agent.add_selected_dice(selected, score)
            self.dice = np.delete(self.dice, dice_indices)
            self.remaining_dice = len(self.dice)

            if self.remaining_dice == 0:
                self.agent.hot_dice = True
                self.remaining_dice = self.n_dice
                self.dice = self._roll_dice(self.n_dice)
            reward = min(1.0, self.agent.current_score / self.score_to_win)

        return self.get_state(), reward, False, {}

    def get_state(self) -> np.ndarray:
        dice_state = np.zeros(6)
        if self.dice is not None:
            dice_state[:len(self.dice)] = self.dice

        selected_dice_state = np.zeros(6)
        if len(self.agent.selected_dice) > 0:
            values, counts = np.unique(
                self.agent.selected_dice, return_counts=True)
            for v, c in zip(values, counts):
                selected_dice_state[int(v) - 1] = c

        state = np.concatenate([
            dice_state,                     # Dice (6)
            selected_dice_state,            # selected dice (6)
            [self.agent.current_score],     # Current turn score (1)
            [self.agent.total_score],       # Agent total score (1)
            [self.opponent.total_score],    # Opponent total score (1)
            [self.remaining_dice],          # remaining dice (1)
            [float(self.agent.hot_dice)],   # hot dice flag (1)
        ])

        return state

    @property
    def action_space(self) -> int:
        return {
            'n': 8,
            'values': list(range(8))
        }

    @property
    def observation_space(self) -> int:
        return {
            'shape': (17,),
            'low': np.array([0] * 17),
            'high': np.array([6] * 12 + [self.score_to_win] * 3 + [6, 1])
        }

    def render(self, mode: str = 'human') -> None:
        if mode == 'human':
            print(f"\nAvailable Dice: {self.dice}")
            print(f"Selected Dice: {self.agent.selected_dice}")
            print(f"Current Score: {self.agent.current_score}")
            print(f"Agent Total Score: {self.agent.total_score}")
            print(f"Opponent Total Score: {self.opponent.total_score}")
            print(f"Remaining Dice: {self.remaining_dice}")
            print(f"Hot Dice: {self.agent.hot_dice}")


def play_farkle_vs_random():
    env = Farkle()

    state = env.reset()
    done = False

    while not done:
        env.render()
        print("Please select an action:")
        print("0: Roll")
        print("1: Bank")
        print("2-7: Keep dice")
        action = int(input())
        state, reward, done, info = env.step(action)
