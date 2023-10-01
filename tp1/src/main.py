import logging
from enum import Enum
from typing import Callable
from itertools import product
from dataclasses import dataclass

import gymnasium
import numpy as np

logging.basicConfig(
    format='%(asctime)s | %(levelname)-8s | %(name)s : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)


class Action(Enum):
    STICK: int = 0
    HIT: int = 1

    @classmethod
    def mapper(cls, value: [int, float]):
        return {0: cls.STICK, 1: cls.HIT}[int(value)]


class Reward(Enum):
    WIN: int = +1
    WIN_NATURAL: int = +2
    LOSS: int = -1
    DRAW: int = +0

    @classmethod
    def from_env(cls, value: [int, float]):
        """
        win game: +1
        lose game: -1
        draw game: 0
        win game with natural blackjack: +1.5 (if natural is True) +1 (if natural is False)
        """
        mapper = {
            1: cls.WIN,
            1.5: cls.WIN_NATURAL,
            -1: cls.LOSS,
            0: cls.DRAW,
        }

        return mapper[value]


@dataclass
class State:
    CSUM: int
    CARDV: int
    ACE: int

    def __post_init__(self):
        if self.CSUM > 21:
            self.CSUM = 22


def eps_greedy(values, eps: float) -> Action:
    action_best = np.argmax(values)

    p1 = 1.0 - eps + eps / len(Action)
    p2 = eps / len(Action)
    prob = [p1 if a == action_best else p2 for a in range(len(Action))]

    return Action.mapper(np.random.choice(len(Action), p=np.array(prob)))


class BlackjackQTable:
    table_: np.ndarray

    def __init__(self, alpha: float, gamma: float):
        self.logger = logging.getLogger('BlackjackQTable')

        self.alpha = alpha
        self.gamma = gamma

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
            self.table_ = np.random.random(product(*self.shape)).reshape(self.shape)
        else:
            raise ValueError(f'init value {init} is not defined')

        return self

    def get(self, state: State) -> np.ndarray:
        return self.table_[state.CSUM, state.CARDV, state.ACE, :]

    def update(self, statec: State, staten: State, action: Action, reward: Reward) -> None:
        self.table_[statec.CSUM, statec.CARDV, statec.ACE, action.value] = (
            self.table_[statec.CSUM, statec.CARDV, statec.ACE, action.value]
            + self.alpha * (
                reward.value
                + self.gamma * self.table_[staten.CSUM, statec.CARDV, statec.ACE, :].max()
                - self.table_[statec.CSUM, statec.CARDV, statec.ACE, action.value]
            )
        )

        self.updates += 1

        count_nonzero = np.sum(list(map(len, np.where(self.table_ > 0))))
        self.logger.debug(f'nonzero qtable values: {count_nonzero}')
        self.logger.debug(f'current state: {statec}')
        self.logger.debug(f'action value: {self.table_[staten.CSUM, statec.CARDV, statec.ACE, :]}')

class BlackjackLearn(object):
    def __init__(self, max_iter: int, max_actions: int, policy: Callable, qtable: BlackjackQTable):
        self.logger = logging.getLogger('BlackjackLearn')

        self.max_iter = max_iter
        self.max_actions = max_actions
        self.policy = policy
        self.qtable = qtable

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
            observation, reward, terminated, truncated, _ = self.env.step(action.value)
            self.qtable.update(
                statec=state,
                staten=State(*observation),
                action=action,
                reward=Reward.from_env(reward),
            )

            state = State(*observation)
            action = self.policy(self.qtable.get(state))
            rewards.append(Reward.from_env(reward))

            if terminated or truncated:
                break

        return rewards

    def run_episode(self):
        rewards = list()
        for i in range(self.max_iter):
            self.logger.debug(f'run episode {i:05d}')
            reward = self.run_iteration()
            rewards.append(np.sum(list(map(lambda x: x.value, reward))))

        return rewards

    def learn(self):
        self.logger.info('start learning')
        rewards = self.run_episode()
        self.logger.debug('finish learning')
        return np.sum(np.array(rewards))

    def __del__(self):
        self.env.close()


def main():
    logger = logging.getLogger('main')

    learner = BlackjackLearn(
        max_iter=32, max_actions=32,
        policy=lambda x: eps_greedy(values=x, eps=0.5),
        qtable=BlackjackQTable(gamma=0.5, alpha=0.5).set(init='zero'),
    )

    n_eps = 2000
    for i in range(n_eps):
        reward = learner.run_episode()
        logger.info(f'episode {i} : +reward {len(np.where(np.array(reward) > 0)[0])}')


if __name__ == '__main__':
    main()
