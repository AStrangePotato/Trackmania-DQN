import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

class ActorCritic(nn.Module):
    def __init__(self, action_dim):
        super(ActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self._init_conv_output()
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(512, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(512, 1)

    def _init_conv_output(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, 4, 96, 96)
            conv_out = self.conv(sample_input)
            self.conv_output_size = conv_out.view(1, -1).size(1)

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def act(self, state):
        x = self.forward(state)
        mean = self.actor_mean(x)
        std = self.actor_log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        value = self.critic(x).squeeze(1)
        return action, log_prob, value

    def evaluate(self, states, actions):
        x = self.forward(states)
        mean = self.actor_mean(x)
        std = self.actor_log_std.exp()
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        values = self.critic(x).squeeze(1)
        return log_probs, entropy, values

class PPO:
    def __init__(self, action_dim, lr=2.5e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.policy = ActorCritic(action_dim).cuda()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return returns

    def update(self, memory):
        states = torch.stack(memory.states).cuda()
        actions = torch.stack(memory.actions).cuda()
        old_log_probs = torch.stack(memory.log_probs).cuda()
        returns = torch.tensor(memory.returns).cuda()
        values = torch.stack(memory.values).cuda()
        advantages = returns - values

        for _ in range(self.k_epochs):
            log_probs, entropy, new_values = self.policy.evaluate(states, actions)
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() + 0.5 * (returns - new_values).pow(2).mean() - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
