import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
import os

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.returns = []

    def store(self, state, action, logprob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.returns = []

class PPO(nn.Module):
    def __init__(self, input_dim, action_dim, learning_rate=1e-4, gamma=0.995, gae_lambda=0.95, 
                 clip_coef=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, 
                 update_epochs=4, minibatch_size=256, norm_adv=True, clip_vloss=True):
        super(PPO, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss

        # Initialize actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        self._init_weights(self.actor, std=0.01)

        # Initialize critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self._init_weights(self.critic, std=1.0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)
        
        # Logging setup
        self.run_name = f"TrackMania_PPO_{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.global_step = 0
        self.start_time = time.time()

    def save_checkpoint(self, checkpoint_dir="checkpoints", filename=None):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if filename is None:
            filename = f"ppo_checkpoint_{self.global_step}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.global_step = checkpoint['global_step']
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path} at step {self.global_step}")

    def _init_weights(self, module, std):
        for layer in module:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, std)
                nn.init.constant_(layer.bias, 0.0)

    def act(self, state):
        state = state.to(self.device)
        logits = self.actor(state)
        probs = Categorical(logits=logits)
        action = probs.sample()
        logprob = probs.log_prob(action)
        value = self.critic(state)
        return action, logprob, value.squeeze()

    def get_value(self, state):
        state = state.to(self.device)
        return self.critic(state).squeeze()

    def compute_gae(self, rewards, dones, values, next_value):
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        next_value = torch.tensor(next_value, dtype=torch.float32).to(self.device)

        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
        return returns

    def update(self, memory):
        self.global_step += len(memory.states)

        # Convert memory to tensors
        states = torch.stack(memory.states).to(self.device)
        actions = torch.tensor(memory.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.tensor(memory.logprobs).to(self.device)
        returns = memory.returns.to(self.device)
        values = torch.tensor(memory.values).to(self.device)

        advantages = returns - values
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy and value network
        batch_size = len(memory.states)
        indices = np.arange(batch_size)
        clipfracs = []

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_logprobs = old_logprobs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_values = values[mb_indices]

                _, new_logprobs, entropy, new_values = self._get_action_and_value(mb_states, mb_actions)
                
                logratio = new_logprobs - mb_old_logprobs
                ratio = logratio.exp()

                # Calculate approximate KL
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_values = new_values.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (new_values - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(new_values - mb_values, -self.clip_coef, self.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Early stopping on KL divergence
            if approx_kl > 0.02:  # Reasonable default for target_kl
                print("KL LOSS EXCEEDED")
                break

        # Logging
        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
        self.writer.add_scalar("charts/reward", np.mean(memory.rewards), self.global_step)

    def _get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)