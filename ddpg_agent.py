import copy
import random
from collections import deque, namedtuple
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from actor_network import ActorNetwork
from critic_network import CriticNetwork
from noise import OUNoise
from replay_buffer import ReplayBuffer
from torch import nn

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 5e-4  # learning rate of the actor
LR_CRITIC = 5e-4  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay
EPSILON = 1.0  # epsilon start
UPDATE_EVERY = 20  # update every step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    """Interacts with and learns from the environment."""
    def __init__(self, state_size: int, action_size: int, random_seed: int):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size

        # Actor Network (w/ Target Network)
        self.actor_local = ActorNetwork(state_size, action_size,
                                        random_seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size,
                                         random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = CriticNetwork(state_size, action_size,
                                          random_seed).to(device)
        self.critic_target = CriticNetwork(state_size, action_size,
                                           random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE,
                                   random_seed, device)
        self.t_step = 0
        self.random = random.Random(random_seed)
        self.eps = 1.0
        self.eps_decay = 0.9995
        self.eps_end = 0.01

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(
                states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if len(self.memory) > BATCH_SIZE:
            if self.t_step % UPDATE_EVERY == 0:
                for _ in range(10):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, states, add_noise: bool = True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            for a in action:
                a += self.eps * self.noise.sample()
            self.eps = max(self.eps_decay * self.eps, self.eps_end)
        action = np.clip(action, -1.0, 1.0)

        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences: Tuple, gamma: float):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model: nn.Module, target_model: nn.Module,
                    tau: float):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)
