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
TD_ERROR_EPSILON = 0.0001


def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[0] / 72.0, frames[0].shape[0] / 72.0))
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50
    )

    anim.save("gif/movie_cartpole_prioritized_experience_replay.gif")


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

    def update_Q_function(self):
        self.brain.replay()

    def get_action(self, state, step):
        action = self.brain.decide_action(state, step)
        return action

    def memories(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_network(self):
        self.brain.update_target_q_network()


class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)

        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)
        self.target_q_network = Net(n_in, n_mid, n_out)
        print(self.main_q_network)

        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.0001
        )

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        (
            self.batch,
            self.state_batch,
            self.action_batch,
            self.reward_batch,
            self.non_final_next_states,
        ) = self.make_minibatch()

        self.expected_state_action_values = (
            self.get_expected_state_action_values()
        )

        self.update_main_q_network()

    def update_main_q_network(self):
        self.main_q_network.train()

        loss = F.smooth_l1_loss(
            self.state_action_values,
            self.expected_state_action_values.unsqueeze(1),
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

    def make_minibatch(self):
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        return (
            batch,
            state_batch,
            action_batch,
            reward_batch,
            non_final_next_states,
        )

    def get_expected_state_action_values(self):
        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(
            self.state_batch
        ).gather(1, self.action_batch)

        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, self.batch.next_state))
        )
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        a_m[non_final_mask] = (
            self.main_q_network(self.non_final_next_states).max(1)[1].detach()
        )

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = (
            self.target_q_network(self.non_final_next_states)
            .gather(1, a_m_non_final_next_states)
            .detach()
            .squeeze()
        )
        expected_state_action_values = (
            self.reward_batch + GAMMA * next_state_values
        )
        return expected_state_action_values

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
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
        episode_10_list = np.zeros(10)
        complete_episodes = 0
        is_episode_final = False
        frames = []

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()

            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)

            for step in range(MAX_STEPS):
                if is_episode_final is True:
                    frames.append(self.env.render(mode="rgb_array"))

                action = self.agent.get_action(state, episode)

                observation_next, _, done, _ = self.env.step(action.item())

                if done:
                    state_next = None
                    episode_10_list = np.hstack(
                        (episode_10_list[1:], step + 1)
                    )
                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes += 1
                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(
                        torch.FloatTensor
                    )
                    state_next = torch.unsqueeze(state_next, 0)

                self.agent.memories(state, action, state_next, reward)
                self.agent.update_Q_function()

                state = state_next

                if done:
                    print(
                        f"{episode} Episode: Finished after {step + 1} steps"
                    )
                    if episode % 2 == 0:
                        self.agent.update_target_q_network()
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
