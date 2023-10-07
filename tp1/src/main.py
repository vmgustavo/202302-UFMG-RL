import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from src.blackjack import BlackjackLearn, Reward
from src.qlearning import BlackjackQTable


@hydra.main(version_base=None, config_path='conf', config_name='blackjack')
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger('main')

    outdir = Path(HydraConfig.get().runtime.output_dir)

    qtable = (
        BlackjackQTable(
            gamma=0.99, alpha=0.1,
            shape=(
                32,  # total sum of cards
                12,  # maximum value for a card
                2,   # usable ace
                2,   # two possible actions
            )
        )
        .set(init='runif')
    )

    learner = BlackjackLearn(
        max_iter=8,
        qtable=qtable,
        rewardspec=Reward(**cfg.rewardspec)
    )

    with plt.ion():
        plt.figure(figsize=(10, 10))
        plt.title('Q-Table')
        rewards_episode = list()
        for i in range(cfg.n_episodes + 1):
            reward = learner.run_episode()
            rewards_episode.append(reward)
            logger.info(f'episode {i} : episode reward {reward}')

            if i % 50 == 0:
                qtable.plot(action='hit')

        with open(outdir / f'episodes_{cfg.n_episodes}__rewards.csv', 'w') as f:
            f.writelines([f'{elem}\n' for elem in rewards_episode])

    qtable.dump(path=outdir / f'episodes_{cfg.n_episodes}__BlackjackQTable.pickle')

    if cfg.interactive.plot:
        qtable.plot(action='hit', savefig_path=outdir / f'episodes_{cfg.n_episodes}__BlackjackQTable.png')


if __name__ == '__main__':
    main()
