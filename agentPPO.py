import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from collections import deque

# === PPO Hyperparameters === #
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPS = 0.2
BATCH_SIZE = 64
EPOCHS = 4
LR = 2e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 5

STATE_DIM = 7
ACTION_DIM = 6

# === Neural Net for Actor-Critic === #
class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        # Two hidden layers of 64 neurons each
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = self.shared(state)
        return self.policy_head(x), self.value_head(x)

    def act(self, state):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy(), value.squeeze()

# === Agent and Trajectory Buffer === #
class PPOAgent:
    def __init__(self, train_freq=1024):
        self.model = ActorCritic(STATE_DIM, 64, ACTION_DIM).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.buffer = []
        self.total_steps = 0
        self.train_freq = train_freq  # ← train every N steps

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
        with torch.no_grad():
            action, log_prob, entropy, value = self.model.act(state_tensor)
        return action, log_prob.cpu(), entropy.cpu(), value.cpu()

    def remember(self, transition):
        self.buffer.append(transition)
        self.total_steps += 1

    def ready_to_train(self):
        return len(self.buffer) >= self.train_freq

    def compute_gae(self, rewards, values, dones):
        advantages, gae = [], 0.0
        values = values + [0.0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def train(self):
        if not self.ready_to_train():
            return  # Not enough data yet or insufficient data for a batch
        
        print(f"Training on {len(self.buffer)} steps")
        states, actions, old_log_probs, rewards, values, dones = zip(*self.buffer)
        states_tensor = torch.tensor(states, dtype=torch.float32).cuda()
        actions_tensor = torch.tensor(actions).cuda()
        old_log_probs = torch.stack(old_log_probs).detach().cuda()
        rewards = list(rewards)
        values = [v.item() for v in values]
        dones = [int(d) for d in dones]

        advantages, returns = self.compute_gae(rewards, values, dones)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32).cuda()
        ret_tensor = torch.tensor(returns, dtype=torch.float32).cuda()
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        for _ in range(EPOCHS):
            idxs = np.arange(len(self.buffer))
            np.random.shuffle(idxs)
            for start in range(0, len(self.buffer), BATCH_SIZE):
                batch_idx = idxs[start:start + BATCH_SIZE]
                b_states = states_tensor[batch_idx]
                b_actions = actions_tensor[batch_idx]
                b_old_log = old_log_probs[batch_idx]
                b_returns = ret_tensor[batch_idx]
                b_adv = adv_tensor[batch_idx]

                logits, value_pred = self.model(b_states)
                dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
                log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - b_old_log)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - PPO_EPS, 1 + PPO_EPS) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value_pred.view(-1), b_returns)
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        self.buffer.clear()  # clear buffer after training

    def save(self, path="model/ppo_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="model/ppo_model.pth"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()