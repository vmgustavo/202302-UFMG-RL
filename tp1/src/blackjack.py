import logging
from enum import Enum
from dataclasses import dataclass

import gymnasium
import numpy as np
from numba import njit

from src.utils import random_choice


class Action(Enum):
    STICK: int = 0
    HIT: int = 1

    @classmethod
    def mapper(cls, value: [int, float]):
        return {0: cls.STICK, 1: cls.HIT}[int(value)]


@dataclass
class State:
    CSUM: int
    CARDV: int
    ACE: int


@dataclass(frozen=True)
class Reward:
    WIN: int
    WIN_NATURAL: int
    LOSS: int
    DRAW: int

    def from_env(self, outcome: [int, float]) -> int:
        """
        win game: +1
        lose game: -1
        draw game: 0
        win game with natural blackjack: +1.5 (if natural is True) +1 (if natural is False)
        """
        mapper = {
            1: self.WIN,
            1.5: self.WIN_NATURAL,
            -1: self.LOSS,
            0: self.DRAW,
        }

        return mapper[outcome]


@njit
def eps_greedy(values, eps: float, nactions: int) -> Action:
    action_best = np.argmax(values)

    p1 = 1.0 - eps + eps / nactions
    p2 = eps / nactions
    prob = [p1 if a == action_best else p2 for a in range(nactions)]

    return random_choice(np.array(prob))


class BlackjackLearn:
    def __init__(self, max_iter: int, qtable, rewardspec: Reward):
        self.logger = logging.getLogger('BlackjackLearn')

        self.max_iter = max_iter
        self.policy = lambda x: Action.mapper(eps_greedy(values=x, eps=1E-2, nactions=len(Action)))
        self.qtable = qtable
        self.rewardspec = rewardspec

        self.env = gymnasium.make('Blackjack-v1', natural=False, sab=True)
        self.actions = Action
        self.states = State

        self.rewards = list()

    def run_episode(self):
        c_observation, info = self.env.reset()
        c_state = State(*c_observation)

        reward_episode_total = 0
        for i in range(self.max_iter):
            c_action = self.policy(self.qtable.get(c_state))

            n_observation, c_outcome, terminated, truncated, _ = self.env.step(c_action.value)
            self.qtable.update(
                c_state=c_state,
                n_state=State(*n_observation),
                c_action=c_action,
                reward=self.rewardspec.from_env(c_outcome),
            )

            c_state = State(*n_observation)

            reward_episode_total += self.rewardspec.from_env(c_outcome)

            if terminated or truncated:
                break

        return reward_episode_total

    def __del__(self):
        self.env.close()

