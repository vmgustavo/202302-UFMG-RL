import gymnasium


def main():
    # https://gymnasium.farama.org/environments/atari/breakout/
    print()
    env = gymnasium.make('ALE/Breakout-v5')
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        print(reward, terminated, truncated, info)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == '__main__':
    main()
