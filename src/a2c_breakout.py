import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
from gym import spaces
from gym.spaces.box import Box
from baselines.common import atari_wrappers

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import cv2

cv2.ocl.setUseOpenCL(False)

ENV_NAME = "BreakoutNoFrameskip-v4"
NUM_SKIP_FRAME = 4
NUM_STACK_FRAME = 4
NOOP_MAX = 30
NUM_PROCESSES = 16
NUM_ADVANCED_STEP = 5
GAMMA = 0.99
TOTAL_FRAMES = 10e6
NUM_UPDATES = int(TOTAL_FRAMES / NUM_ADVANCED_STEP / NUM_PROCESSES)

VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

LR = 7e-4
EPS = 1e-5
ALPHA = 0.99

use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")


class WrapFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame[:, :, None]


class WrapPytorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPytorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env = atari_wrappers.NoopResetEnv(env, noop_max=30)
        env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
        env.seed(seed + rank)
        env = atari_wrappers.EpisodicLifeEnv(env)
        env = WrapFrame(env)
        env = WrapPytorch(env)
        return env

    return _thunk


class RolloutStorage:
    def __init__(self, num_steps, num_processes, obs_shape):
        self.observations = torch.zeros(
            num_steps + 1, num_processes, *obs_shape
        ).to(DEVICE)
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(DEVICE)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(DEVICE)
        self.actions = (
            torch.zeros(num_steps, num_processes, 1).long().to(DEVICE)
        )

        self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(DEVICE)
        self.index = 0

    def insert(self, current_obs, action, reward, mask):
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = (
                self.returns[ad_step + 1] * GAMMA * self.masks[ad_step + 1]
                + self.rewards[ad_step]
            )


def init(module, gain):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, n_out):
        super(Net, self).__init__()

        def init_(module):
            return init(module, gain=nn.init.calculate_gain("relu"))

        self.conv = nn.Sequential(
            init_(nn.Conv2d(NUM_STACK_FRAME, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        def init_(module):
            return init(module, gain=1.0)

        self.critic = init_(nn.Linear(512, 1))

        def init_(module):
            return init(module, gain=0.01)

        self.actor = init_(nn.Linear(512, n_out))

        self.train()

    def forward(self, x):
        input = x / 255.0
        conv_output = self.conv(input)
        critic_output = self.critic(conv_output)
        actor_output = self.actor(conv_output)
        return critic_output, actor_output

    def act(self, x):
        value, actor_output = self(x)
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)
        return action

    def get_value(self, x):
        value, _ = self(x)
        return value

    def evaluate_actions(self, x, actions):
        value, actor_output = self(x)
        log_probs = F.log_softmax(actor_output, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        probs = F.softmax(actor_output, dim=1)
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy


class Brain:
    def __init__(self, actor_critic: Net):
        self.actor_critic = actor_critic
        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr=LR, eps=EPS, alpha=ALPHA
        )

    def update(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, 1),
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values

        value_loss = advantages.pow(2).mean()

        action_gain = (action_log_probs * advantages.detach()).mean()

        total_loss = (
            value_loss * VALUE_LOSS_COEF - action_gain - entropy * ENTROPY_COEF
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()


class Environment:
    def run(self):

        seed_num = 1
        torch.manual_seed(seed_num)
        if use_cuda:
            torch.cuda.manual_seed(seed_num)

        torch.set_num_threads(seed_num)
        envs = [make_env(ENV_NAME, seed_num, i) for i in range(NUM_PROCESSES)]
        envs = SubprocVecEnv(envs)

        n_out = envs.action_space.n
        actor_critic = Net(n_out).to(DEVICE)
        global_brain = Brain(actor_critic)

        obs_shape = envs.observation_space.shape
        obs_shape = (obs_shape[0] * NUM_STACK_FRAME, *obs_shape[1:])
        print(obs_shape)
        current_obs = torch.zeros(NUM_PROCESSES, *obs_shape).to(DEVICE)
        print(current_obs.shape)
        rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])
        final_rewards = torch.zeros([NUM_PROCESSES, 1])

        obs = envs.reset()
        obs = torch.from_numpy(obs).float()
        current_obs[:, -1:] = obs

        rollouts.observations[0].copy_(current_obs)

        for j in tqdm(range(NUM_UPDATES)):

            for step in range(NUM_ADVANCED_STEP):
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])
                cpu_actions = action.squeeze(1).cpu().numpy()

                obs, reward, done, info = envs.step(cpu_actions)

                reward = np.expand_dims(np.stack(reward), 1)
                reward = torch.from_numpy(reward).float()
                episode_rewards += reward

                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]
                )
                final_rewards *= masks

                final_rewards += (1 - masks) * episode_rewards

                episode_rewards *= masks

                masks = masks.to(DEVICE)

                current_obs *= masks.unsqueeze(2).unsqueeze(2)

                obs = torch.from_numpy(obs).float()
                current_obs[:, :-1] = current_obs[:, 1:]

                current_obs[:, -1:] = obs

                rollouts.insert(current_obs, action.data, reward, masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]
                ).detach()

            rollouts.compute_returns(next_value)
            global_brain.update(rollouts)
            rollouts.after_update()

            if j % 100 == 0:
                print(
                    f"finished frames {j*NUM_PROCESSES*NUM_ADVANCED_STEP}"
                    f"mean reward {final_rewards.mean()}"
                    f"median reward {final_rewards.median()}"
                )
            if j % 12500 == 0:
                torch.save(
                    global_brain.actor_critic.state_dict(),
                    "weight_" + str(j) + ".pth",
                )
        torch.save(global_brain.actor_critic.state_dict(), "weight_end.pth")


def main():
    cartpole_env = Environment()
    cartpole_env.run()


if __name__ == "__main__":
    main()
