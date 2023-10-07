import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from src.qlearning import BlackjackQTable
from src.blackjack import BlackjackLearn, Reward, Action


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

            if i % cfg.interactive.step == 0:
                qtable.plot(action=Action.HIT)

        with open(outdir / f'blackjack__rewards.csv', 'w') as f:
            f.writelines([f'{elem}\n' for elem in rewards_episode])

    qtable.dump(path=outdir / 'blackjack__qtable_values.pickle')

    if cfg.interactive.plot:
        qtable.plot(
            action=Action.HIT,
            savefig_path=outdir / f'blackjack__qtable__plot_{Action.HIT}.png'
        )


if __name__ == '__main__':
    main()
