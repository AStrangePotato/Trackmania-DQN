import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

loss_plot = []

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden1_size)
        self.linear2 = nn.Linear(hidden1_size, hidden2_size)
        self.linear3 = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        print("Saved model at", file_name)


class QTrainer:
    def __init__(self, model, target_model, lr, gamma, TAU):
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.TAU = TAU
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.episodes = 0

    def update_target_model(self):
        td = self.target_model.state_dict()
        pd = self.model.state_dict()
        for k in pd:
            td[k] = pd[k]*self.TAU + td[k]*(1-self.TAU)
        self.target_model.load_state_dict(td)

    def cudafy_tensors(self, state, action, reward, next_state):
        return (torch.tensor(state, dtype=torch.float32).cuda(),
                torch.tensor(action, dtype=torch.long).cuda(),
                torch.tensor(reward, dtype=torch.float32).cuda(),
                torch.tensor(next_state, dtype=torch.float32).cuda())

    def train_step(self, state, action, reward, next_state, done):
        state, action, reward, next_state = self.cudafy_tensors(state, action, reward, next_state)

        # batchify single samples
        if state.dim() == 1:
            state      = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action     = action.unsqueeze(0)
            reward     = reward.unsqueeze(0)
            done       = torch.tensor([done], dtype=torch.float32, device=state.device)
        else:
            done = torch.tensor(done, dtype=torch.float32, device=state.device)

        # 1) Current Q estimates and clone as targets
        pred_Q   = self.model(state)                     # (B, n_actions)
        target_Q = pred_Q.clone().detach()               # (B, n_actions)

        # 2) Compute Q_new = r + γ * max Q(next) * (1 - done)
        next_max_Q = self.target_model(next_state).max(dim=1).values  # (B,)
        Q_new = reward + self.gamma * next_max_Q * (1 - done)         # (B,)

        # 3) Insert Q_new into the target Q matrix at the chosen action indices
        chosen = action.argmax(dim=1)                               # (B,)
        batch_idx = torch.arange(state.size(0), device=state.device)
        target_Q[batch_idx, chosen] = Q_new

        # 4) Optimize
        self.optimizer.zero_grad()
        loss = self.criterion(pred_Q, target_Q)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.update_target_model()

        loss_plot.append(loss.item())
