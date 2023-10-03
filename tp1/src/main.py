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

    def __post_init__(self):
        if self.CSUM > 21:
            self.CSUM = 22


@njit
def eps_greedy(values, eps: float, nactions: int) -> Action:
    action_best = np.argmax(values)

    p1 = 1.0 - eps + eps / nactions
    p2 = eps / nactions
    prob = [p1 if a == action_best else p2 for a in range(nactions)]

    return random_choice(np.array(prob))


@njit
def qlearn_update(value_curr, reward_value, gamma, alpha, nextaction):
    temporal_diff = (
            reward_value
            + gamma * nextaction
            - value_curr
    )

    value_new = (
            value_curr
            + alpha * temporal_diff
    )

    return value_new


class BlackjackQTable:
    table_: np.ndarray
    ocurrences_: np.ndarray

    def __init__(self, alpha: float, gamma: float):
        self.logger = logging.getLogger('BlackjackQTable')

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor

        n_csum = 23
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

    def update(self, statec: State, staten: State, action: Action, reward: float) -> None:

        self.table_[statec.CSUM, statec.CARDV, statec.ACE, action.value] = qlearn_update(
            value_curr=self.table_[statec.CSUM, statec.CARDV, statec.ACE, action.value],
            reward_value=reward, gamma=self.gamma, alpha=self.alpha,
            nextaction=self.table_[staten.CSUM, staten.CARDV, staten.ACE, :].max()
        )

        self.updates += 1
        self.ocurrences_[statec.CSUM, statec.CARDV, statec.ACE, action.value] += 1

        count_nonzero = np.sum(list(map(len, np.where(self.table_ > 0))))
        self.logger.debug(f'nonzero qtable values: {count_nonzero}')
        self.logger.debug(f'current state: {statec}')
        self.logger.debug(f'action value: {self.table_[staten.CSUM, statec.CARDV, statec.ACE, :]}')

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
    def __init__(self, max_iter: int, max_actions: int, policy: Callable, qtable: BlackjackQTable, rewardspec: Reward):
        self.logger = logging.getLogger('BlackjackLearn')

        self.max_iter = max_iter
        self.max_actions = max_actions
        self.policy = policy
        self.qtable = qtable
        self.rewardspec = rewardspec

        self.env = gymnasium.make('Blackjack-v1')
        self.actions = Action
        self.states = State

        self.rewards = list()

    def run_iteration(self):
        observation, info = self.env.reset()
        state = State(*observation)
        action = self.policy(self.qtable.get(state))

        rewards = list()
        for i in range(self.max_actions):
            self.logger.debug(f'run action {i:03d}')
            observation, outcome, terminated, truncated, _ = self.env.step(action.value)
            self.qtable.update(
                statec=state,
                staten=State(*observation),
                action=action,
                reward=self.rewardspec.from_env(outcome),
            )

            state = State(*observation)
            action = self.policy(self.qtable.get(state))
            rewards.append(self.rewardspec.from_env(outcome))

            if terminated or truncated:
                break

        return rewards

    def run_episode(self):
        rewards = list()
        for i in range(self.max_iter):
            self.logger.debug(f'run episode {i:05d}')
            reward = self.run_iteration()
            rewards.append(np.sum(reward))

        return rewards

    def learn(self):
        self.logger.info('start learning')
        rewards = self.run_episode()
        self.logger.debug('finish learning')
        return np.sum(np.array(rewards))

    def __del__(self):
        self.env.close()


@hydra.main(version_base=None, config_path='conf', config_name='blackjack')
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger('main')

    outdir = Path(HydraConfig.get().runtime.output_dir)

    qtable = BlackjackQTable(gamma=0.99, alpha=0.5).set(init='runif')
    learner = BlackjackLearn(
        max_iter=256, max_actions=10,
        policy=lambda x: Action.mapper(eps_greedy(values=x, eps=1E-2, nactions=len(Action))),
        qtable=qtable, rewardspec=Reward(**cfg.rewardspec)
    )

    with plt.ion():
        file_rewards = outdir / f'episodes_{cfg.n_episodes}__rewards.csv'
        with open(file_rewards, 'w') as f:
            f.write('')

        for i in range(cfg.n_episodes + 1):
            reward = learner.run_episode()
            logger.info(f'episode {i} : positive reward {len(np.where(np.array(reward) > 0)[0])}')

            with open(file_rewards, 'a') as f:
                f.write(','.join(map(str, reward)) + '\n')

            if i % 50 == 0:
                qtable.plot(action=Action.HIT)

    qtable.dump(path=outdir / f'episodes_{cfg.n_episodes}__BlackjackQTable.pickle')


if __name__ == '__main__':
    main()
