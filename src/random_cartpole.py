import numpy as np
import matplotlib.pyplot as plt
import gym

from matplotlib import animation


def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[0] / 72.0, frames[0].shape[0] / 72.0))
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50
    )

    anim.save("movie_cartpole.gif")
    # plt.show()
    # display(display_animation(anim, default_mode="loop"))


def random_move():
    frames = []
    env = gym.make("CartPole-v0")
    observation = env.reset()

    for step in range(200):
        frames.append(env.render(mode="rgb_array"))
        action = np.random.choice(2)
        observation, reward, done, info = env.step(action)
    return frames


def main():
    frames = random_move()
    display_frames_as_gif(frames)


if __name__ == "__main__":
    main()
