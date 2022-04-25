from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.use_cuda = False #torch.cuda.is_available()

    def train_step(self, state, action, reward, next_state, is_done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            is_done = (is_done, )
        
        if self.use_cuda:
            state = state.cuda()
            action = action.cuda()
            reward = reward.cuda()
            next_state = next_state.cuda()
            
        # 1: predicted Q values with current state
        prediction = self.model(state)
        target = prediction.clone()
        for index in range(len(is_done)):
            Q_new = reward[index]
            if not is_done[index]:
                Q_new = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
                # print(Q_new, reward[index], torch.max(self.model(next_state[index])))

            # print('Q_new', Q_new)
            target[index] = Q_new # I don't know what second index to use, I just used 0

        # 2: Q_new = r + y * max(next predicted Q value) -> only if not done
        # prediction.clone()
        # predictions[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
        