import numpy as np
import matplotlib.pyplot as plt
import gym

from matplotlib import animation

ENV = "Breakout-v0"


def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[0] / 72.0, frames[0].shape[0] / 72.0))
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50
    )

    anim.save("git/movie_breakout_random.gif")


def random_move():
    env = gym.make(ENV)
    frames = []
    observation = env.reset()

    for step in range(1000):
        frames.append(observation)
        action = np.random.randint(0, 4)
        observation_next, reward, done, info = env.step(action)

        observation = observation_next

        if done:
            break

    return frames


def main():
    frames = random_move()
    display_frames_as_gif(frames)


if __name__ == "__main__":
    main()
