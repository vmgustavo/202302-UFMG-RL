import pickle
import logging
from enum import Enum
from pathlib import Path
from operator import mul
from typing import Callable
from functools import reduce
from dataclasses import dataclass

import hydra
from hydra.core.hydra_config import HydraConfig
import gymnasium
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from src.utils import random_choice


class Action(Enum):
    STICK: int = 0
    HIT: int = 1

    @classmethod
    def mapper(cls, value: [int, float]):
        return {0: cls.STICK, 1: cls.HIT}[int(value)]


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


@dataclass
class State:
    CSUM: int
    CARDV: int
    ACE: int


@njit
def eps_greedy(values, eps: float, nactions: int) -> Action:
    action_best = np.argmax(values)

    p1 = 1.0 - eps + eps / nactions
    p2 = eps / nactions
    prob = [p1 if a == action_best else p2 for a in range(nactions)]

    return random_choice(np.array(prob))


@njit
def qlearn_update(currvalue, reward_value, gamma, alpha, next_best_action):
    temporal_diff = (
            reward_value
            + gamma * next_best_action
            - currvalue
    )

    value_new = (
            currvalue
            + alpha * temporal_diff
    )

    return value_new


# @njit
# def qlearn_update(currvalue: float, reward_value: float, gamma: float, alpha: float, next_best_action: float):
#     return (
#         (1 - alpha) * currvalue
#         + alpha * (reward_value + gamma * next_best_action)
#     )


class BlackjackQTable:
    table_: np.ndarray
    ocurrences_: np.ndarray

    def __init__(self, alpha: float, gamma: float):
        self.logger = logging.getLogger('BlackjackQTable')

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor

        n_csum = 32
        n_cardv = 12
        n_ace = 2
        n_actions = len(Action)
        self.shape = (n_csum, n_cardv, n_ace, n_actions)

        self.updates = 0

    def set(self, init: str):
        assert init in {'zero', 'runif'}

        if init == 'zero':
            self.table_ = np.zeros(shape=self.shape)
        elif init == 'runif':
            self.table_ = np.random.random(reduce(mul, self.shape)).reshape(self.shape)  # noqa
        else:
            raise ValueError(f'init value {init} is not defined')

        self.ocurrences_ = np.zeros(shape=self.shape)

        return self

    def get(self, state: State) -> np.ndarray:
        return self.table_[state.CSUM, state.CARDV, state.ACE, :]

    def update(self, c_state: State, n_state: State, c_action: Action, reward: float) -> None:

        self.table_[c_state.CSUM, c_state.CARDV, c_state.ACE, c_action.value] = qlearn_update(
            currvalue=self.table_[c_state.CSUM, c_state.CARDV, c_state.ACE, c_action.value],
            reward_value=reward, gamma=self.gamma, alpha=self.alpha,
            next_best_action=self.table_[n_state.CSUM, n_state.CARDV, n_state.ACE, :].max()
        )

        self.updates += 1
        self.ocurrences_[c_state.CSUM, c_state.CARDV, c_state.ACE, c_action.value] += 1

        count_nonzero = np.sum(list(map(len, np.where(self.table_ > 0))))
        self.logger.debug(f'nonzero qtable values: {count_nonzero}')
        self.logger.debug(f'current state: {c_state}')
        self.logger.debug(f'action value: {self.table_[n_state.CSUM, c_state.CARDV, c_state.ACE, :]}')

    def dump(self, path: [str, Path] = 'BlackjackQTable.pickle'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def plot(self, action: Action = Action.HIT):
        plt.gca()
        plt.imshow(self.table_[:, :, 0, action.value])
        plt.ylabel('Current Sum')
        plt.xlabel('Card Value')

        plt.show()
        plt.pause(.1)

    @staticmethod
    def load(path: str = 'BlackjackQTable.pickle'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

        return obj


class BlackjackLearn:
    def __init__(self, max_iter: int, policy: Callable, qtable: BlackjackQTable, rewardspec: Reward):
        self.logger = logging.getLogger('BlackjackLearn')

        self.max_iter = max_iter
        self.policy = policy
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


@hydra.main(version_base=None, config_path='conf', config_name='blackjack')
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger('main')

    outdir = Path(HydraConfig.get().runtime.output_dir)

    qtable = BlackjackQTable(gamma=0.99, alpha=0.1).set(init='runif')
    learner = BlackjackLearn(
        max_iter=8,
        policy=lambda x: Action.mapper(eps_greedy(values=x, eps=1E-2, nactions=len(Action))),
        qtable=qtable, rewardspec=Reward(**cfg.rewardspec)
    )

    with plt.ion():
        rewards_episode = list()
        for i in range(cfg.n_episodes + 1):
            reward = learner.run_episode()
            rewards_episode.append(reward)
            logger.info(f'episode {i} : episode reward {reward}')

            if i % 50 == 0:
                qtable.plot(action=Action.HIT)

        with open(outdir / f'episodes_{cfg.n_episodes}__rewards.csv', 'w') as f:
            f.writelines([f'{elem}\n' for elem in rewards_episode])

    qtable.dump(path=outdir / f'episodes_{cfg.n_episodes}__BlackjackQTable.pickle')


if __name__ == '__main__':
    main()
