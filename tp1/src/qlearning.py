import pickle
import logging
from pathlib import Path
from operator import mul
from functools import reduce

from hydra.core.hydra_config import HydraConfig
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

from src.blackjack import State, Action


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

    def __init__(self, alpha: float, gamma: float, shape: int):
        self.logger = logging.getLogger('BlackjackQTable')

        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.shape = shape

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

    def plot(self, action: str = "hit", savefig_path: str = False):
        plt.gca()
        if action == "hit":
            plt.imshow(self.table_[:, :, 0, Action.HIT.value])
        elif action == "stick":
            plt.imshow(self.table_[:, :, 0, Action.STICK.value])
        plt.ylabel('Current Sum')
        plt.xlabel('Card Value')

        if not savefig_path:
            plt.show()
            plt.pause(.1)
        else:
            plt.savefig(savefig_path)

    @staticmethod
    def load(path: str = 'BlackjackQTable.pickle'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

        return obj

if __name__=="__main__":
    pass
