import math
import json
import random
import logging
import hashlib
from pathlib import Path
from itertools import count
from collections import namedtuple, deque

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig

import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQN(nn.Module):
    def __init__(self, n_observations_, n_actions_):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations_, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions_)

    def forward(self, x):
        x = f.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        return self.layer3(x)


Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model(policy_net, target_net, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


@hydra.main(version_base=None, config_path='conf', config_name='cart-pole')
def main(cfg: DictConfig):
    cbytes = json.dumps(OmegaConf.to_container(cfg, resolve=True)).encode()
    chash = hashlib.sha256(cbytes).hexdigest()
    logger = logging.getLogger(chash[:7])

    outdir = Path(HydraConfig.get().runtime.output_dir)
    logger.info(f'start execution : {outdir}')

    env = gym.make('CartPole-v1')
    n_actions = 2

    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=cfg.model_params.lr, amsgrad=True)
    memory = ReplayMemory(cfg.model_params.memory_size)

    steps_done = 0
    episode_durations = [0]

    if cfg.interactive.plot:
        plt.ion()
        plt.figure(figsize=(12, 5))

    i_episode = 0
    while np.mean(episode_durations[-50:]) < 450:
        i_episode += 1
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            # Select action
            sample = random.random()
            eps_threshold = (
                cfg.model_params.eps_end
                + (cfg.model_params.eps_start - cfg.model_params.eps_end)
                * math.exp(-1. * steps_done / cfg.model_params.eps_decay)
            )
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    action = policy_net(state).max(1).indices.view(1, 1)
            else:
                action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

            # Act on environment
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Evaluate action return
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(
                policy_net=policy_net,
                target_net=target_net,
                memory=memory,
                optimizer=optimizer,
                batch_size=cfg.model_params.batch_size,
                gamma=cfg.model_params.gamma,
            )

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = (
                    policy_net_state_dict[key]
                    * cfg.model_params.tau
                    + target_net_state_dict[key]
                    * (1 - cfg.model_params.tau)
                )
            target_net.load_state_dict(target_net_state_dict)

            if terminated or truncated:
                episode_durations.append(t + 1)
                break
        else:
            eps_threshold = None

        logger.debug(
            f'episode {i_episode}'
            + f' : mean 50 times {np.mean(episode_durations[-50:]):.02f}'
            + f' : eps {eps_threshold:.02f}'
        )

        if cfg.interactive.plot and (i_episode % cfg.interactive.step) == 0:
            n_points = 300
            plt.clf()
            plt.plot(
                range(min(i_episode, n_points) + 1),
                episode_durations[-n_points:]
            )
            plt.plot(
                range(min(i_episode, n_points) + 1),
                (
                    pd.Series(episode_durations)
                    .rolling(window=50, min_periods=1)
                    .mean()
                    .iloc[-n_points:]
                )
            )
            plt.show()
            plt.pause(1E-1)

    with open(outdir / f'cartp__times.csv', 'w') as fout:
        fout.writelines([f'{elem}\n' for elem in episode_durations])


if __name__ == '__main__':
    main()
