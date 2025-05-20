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
        self.lr = lr
        self.gamma = gamma
        self.TAU = TAU
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()
        self.episodes = 1
        
    def update_target_model(self):
        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_model.load_state_dict(target_net_state_dict)

    def cudafy_tensors(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.long).cuda()
        reward = torch.tensor(reward, dtype=torch.float).cuda()
        return state, action, reward, next_state
    
    def train_step(self, state, action, reward, next_state, done):
        state, action, reward, next_state = self.cudafy_tensors(state, action, reward, next_state)

        if len(state.shape) == 1: #add dimension if there is only 1 sample in the batch
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred_Q = self.model(state)
        target_Q = pred_Q.clone() #output array

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_model(next_state[idx]))

            target_Q[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target_Q, pred_Q)
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
        self.optimizer.step()
        self.update_target_model()
        
        loss_plot.append(loss.item())
        




        
