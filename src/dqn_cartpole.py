from collections import namedtuple
import random

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn, optim
import torch.nn.functional as F

from matplotlib import animation

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)
ENV = "CartPole-v0"
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 1000
BATCH_SIZE = 32
CAPACITY = 10000


def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[0] / 72.0, frames[0].shape[0] / 72.0))
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50
    )

    anim.save("movie_cartpole_DQN.gif")


class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_Q_function(self, observation, action, reward, observation_next):
        self.brain.update_Q_table(
            observation, action, reward, observation_next
        )

    def get_action(self, observation, step):
        action = self.brain.decide_action(observation, step)
        return action


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)

        self.model = nn.Sequential()
        self.model.add_module("fc1", nn.Linear(num_states, 32))
        self.model.add_module("relu1", nn.ReLU())
        self.model.add_module("fc2", nn.Linear(32, 32))
        self.model.add_module("relu2", nn.ReLU())
        self.model.add_module("fc3", nn.Linear(32, num_actions))

        print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state])

        self.model.eval()

        state_action_values = self.model(state_batch).gather(1, action_batch)

        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state))
        )
        next_state_values = torch.zeros(BATCH_SIZE)

        next_state_values[non_final_mask] = (
            self.model(non_final_next_states).max(1)[0].detach()
        )

        expected_state_action_values = reward_batch + GAMMA * next_state_values

        self.model.train()

        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
        return action


class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)

    def run(self):
        complete_episodes = 0
        is_episode_final = False
        frames = []

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()

            for step in range(MAX_STEPS):
                if is_episode_final is True:
                    frames.append(self.env.render(mode="rgb_array"))

                action = self.agent.get_action(observation, episode)

                observation_next, _, done, _ = self.env.step(action)

                if done:
                    if step < 195:
                        reward = -1
                        complete_episodes = 0
                    else:
                        reward = 1
                        complete_episodes += 1
                else:
                    reward = 0

                self.agent.update_Q_function(
                    observation, action, reward, observation_next
                )

                observation = observation_next

                if done:
                    print(
                        f"{episode} Episode: Finished after {step + 1} steps"
                    )
                    break

            if is_episode_final:
                display_frames_as_gif(frames)
                break

            if complete_episodes >= 10:
                print("success!!")
                is_episode_final = True


def main():
    cartpole_env = Environment()
    cartpole_env.run()


if __name__ == "__main__":
    main()
