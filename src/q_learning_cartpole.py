import numpy as np
import matplotlib.pyplot as plt
import gym

from matplotlib import animation

ENV = "CartPole-v0"
NUM_DIZITIZED = 6
GAMMA = 0.99
ETA = 0.5
MAX_STEPS = 200
NUM_EPISODES = 1000


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
    env = gym.make(ENV)
    observation = env.reset()

    for step in range(200):
        frames.append(env.render(mode="rgb_array"))
        action = np.random.choice(2)
        observation, reward, done, info = env.step(action)
    return frames


def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


def digitize_state(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, NUM_DIZITIZED)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, NUM_DIZITIZED)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, NUM_DIZITIZED)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, NUM_DIZITIZED)),
    ]
    return sum([x * (NUM_DIZITIZED ** i) for i, x in enumerate(digitized)])


def main():
    frames = random_move()
    display_frames_as_gif(frames)


if __name__ == "__main__":
    main()
