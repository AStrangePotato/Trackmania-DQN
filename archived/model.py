import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

loss_plot = []

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        # He initialization for ReLU networks
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)

        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)

        # Output layer: often use small uniform or zero init
        nn.init.xavier_uniform_(self.fc3.weight)  # or kaiming if using ReLU here too
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        os.makedirs(model_folder_path, exist_ok=True)
        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)
        print("Saved model at", file_path)



class QTrainer:
    def __init__(self, model, target_model, lr, gamma, tau=0.01, target_update_interval=100, l2_reg=1e-5):
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.tau = tau
        self.episodes = 1
        self.target_update_interval = target_update_interval
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5, weight_decay=l2_reg)
        self.criterion = nn.SmoothL1Loss()

    def hard_update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        print(f"[Episode {self.episodes}] Hard updated target network.")

    def train_step(self, state, action, reward, next_state, done):
        device = next(self.model.parameters()).device
        state      = torch.tensor(state, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        action     = torch.tensor(action, dtype=torch.int64, device=device)
        reward     = torch.tensor(reward, dtype=torch.float32, device=device)
        done       = torch.tensor(done, dtype=torch.float32, device=device)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)
        else:
            self.episodes += 1

        # Q(s,a) prediction
        pred_Q = self.model(state)  # shape: [batch_size, num_actions]
        action_indices = action.argmax(dim=1)  # Convert one-hot to indices: [batch_size]
        current_Q = pred_Q.gather(1, action_indices.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Double DQN target: use model to select, target to evaluate
        with torch.no_grad():
            next_action = self.model(next_state).argmax(dim=1, keepdim=True)  # [batch_size, 1]
            next_Q = self.target_model(next_state).gather(1, next_action).squeeze(1)  # [batch_size]
            target_Q = reward + (1 - done) * self.gamma * next_Q

        loss = self.criterion(current_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.episodes % self.target_update_interval == 0:
            self.hard_update_target_model()
            print(f"[Episode {self.episodes}] Loss: {loss.item():.4f} | Q(mean): {pred_Q.mean():.3f}, Q(max): {pred_Q.max():.3f}, Q(min): {pred_Q.min():.3f}")
            loss_plot.append(loss.item())
