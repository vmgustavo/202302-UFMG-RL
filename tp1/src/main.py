from enum import Enum
from typing import Callable
from itertools import product
from dataclasses import dataclass

import gymnasium
import numpy as np


class Action(Enum):
    STICK: int = 0
    HIT: int = 1


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


def eps_greedy(values, eps: float):
    action_best = values.argmax()

    p1 = 1.0 - eps + eps / len(Action)
    p2 = eps / len(Action)
    prob = [p1 if a == action_best else p2 for a in range(len(Action))]

    return np.random.choice(len(Action), p=np.array(prob))


class BlackjackQTable:
    table_: np.ndarray

    def __init__(self, alpha: float, gamma: float):
        self.alpha = alpha
        self.gamma = gamma

        n_csum = 22
        n_cardv = 12
        n_ace = 2
        n_actions = len(Action)
        self.shape = (n_csum, n_cardv, n_ace, n_actions)

    def set(self, init: str):
        assert init in {'zero', 'runif'}

        if init == 'zero':
            self.table_ = np.zeros(shape=self.shape)
        elif init == 'runif':
            self.table_ = np.random.random(product(*self.shape)).reshape(self.shape)
        else:
            raise ValueError(f'init value {init} is not defined')

    def get(self, state: State, policy: Callable):
        values = self.table_[state.CARDV, state.CSUM, state.ACE, :]
        return policy(values)

    def update(self, statec: State, staten: State, action: Action, reward: Reward):
        self.table_[statec.CARDV, statec.CSUM, statec.ACE, action.value] = (
            self.table_[statec.CARDV, statec.CSUM, statec.ACE, action.value]
            + self.alpha * (
                reward.value
                + self.gamma * self.table_[staten.CARDV, statec.CSUM, statec.ACE, :].max()
                - self.table_[statec.CARDV, statec.CSUM, statec.ACE, action.value]
            )
        )


class BlackjackLearn(object):
    def __init__(self, max_iter: int, max_actions: int, policy: Callable, qtable: BlackjackQTable):
        self.max_iter = max_iter
        self.max_actions = max_actions
        self.policy = policy
        self.qtable = qtable

        self.env = gymnasium.make('Blackjack-v1')
        self.actions = Action
        self.states = State

        self.rewards = list()

    def qlearn(self, state: State):
        action = self.policy(state)
        [Sl, R, done, _] = self.env.step(action.value)
        self.qtable[state.CSUM][state.CARDV] = self.qtable[S, A] + self.alpha * (R + self.gamma * self.Q[Sl, :].max() - self.Q[S, A])
        return Sl, R, done

    def run_iteration(self):
        # starting point
        observation, info = self.env.reset()
        state = State(*observation)
        action = self.policy(state)

        for _ in range(self.max_actions):
            observation, reward, terminated, truncated, _ = self.env.step(action.value)
            self.qtable.update(
                statec=state,
                staten=State(*observation),
                action=action,
                reward=Reward.from_env(reward),
            )

        for _ in range(max_iter):
            Sl, R, done = self.qlearning(S)
            S = Sl
            rewards.append(R)
            if done:
                break

    def run_episode(self):
        rewards = list()
        for _ in range(self.max_iter):
            self.run_iteration()

        return rewards

    def learn(self):
        rewards = self.run_episode()
        return np.sum(np.array(rewards))

    def __del__(self):
        self.env.close()


def main():
    learner = BlackjackLearn(
        max_iter=32, max_actions=32,
        policy=lambda x: eps_greedy(values=x, eps=0.5),
        qtable=BlackjackQTable(gamma=0.5, alpha=0.5),
    )

    n_eps = 2000
    rewards = [learner.learn() for i in range(n_eps)]


if __name__ == '__main__':
    main()
