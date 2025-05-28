import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)


class Memory:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.returns = []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)


class PPO:
    def __init__(self, input_dim, action_dim, lr=2.5e-4, gamma=0.99, gae_lambda=0.95, eps_clip=0.2, epochs=4, batch_size=64):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = ActorCritic(input_dim, action_dim).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        logits, value = self.model(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states, actions):
        logits, values = self.model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)

    def compute_gae(self, rewards, dones, values, next_value):
        gae = 0
        returns = []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, memory):
        states = torch.stack(memory.states).cuda()
        actions = torch.tensor(memory.actions).cuda()
        old_log_probs = torch.stack(memory.log_probs).cuda().detach()
        returns = torch.tensor(memory.returns).cuda()
        values = torch.stack(memory.values).cuda().detach()

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            indices = torch.randperm(len(states))
            for i in range(0, len(states), self.batch_size):
                idx = indices[i:i + self.batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]

                log_probs, entropy, values = self.evaluate(batch_states, batch_actions)
                ratio = (log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (batch_returns - values).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()