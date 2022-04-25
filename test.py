import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import numpy as np

width = 5
height = 5

grid = []
for x in range(width):
    grid.append([0] * height)


class Linear_QNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear1_1 = nn.Linear(size, size * 2)
        self.linear1_2 = nn.Linear(size * 2, 2)
    
    def forward(self, x):
        x = F.relu(self.linear1_1(x))
        x = self.linear1_2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

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


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(width * height)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)

    def get_state(self):
        state = []
        for x in range(width):
            for y in range(height):
                state.append(grid[x][y])
                    
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, is_done):
        self.memory.append((state, action, reward, next_state, is_done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, is_dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, is_dones)

    def train_short_memory(self, state, action, reward, next_state, is_done):
        self.trainer.train_step(state, action, reward, next_state, is_done)

    def get_action(self, state):
        # random moves: exploration vs exploitation
        self.epsilon = 80 - self.n_games
        
        if (random.randint(0, 200) < self.epsilon):
            x = random.randint(0, width)
            y = random.randint(0, height)
            prediction = [x, y]
            # print('Random Action', prediction)
            return prediction
        else:
            state0 = torch.tensor(state, dtype=torch.float)

            prediction = self.model(state0)
            # print('"Smart" Action', prediction)
            # TODO: idk 
            ret = [
                int(prediction[0]), 
                int(prediction[1])]

            if (0 <= ret[0] < width and 0 <= ret[1] < height):
                print('Smart Prediction!', ret)
            else:
                print('Dumb Prediction', ret)

            return ret


def perform_action(move):
    x = move[0]
    y = move[1]
    print('Action', x, y)
    score = 0
    for x in range(width):
        for y in range(height):
            if grid[x][y] == 1:
                score += 1

    if (0 <= x < width and 0 <= y < height):
        if grid[x][y] == 0:
            grid[x][y] = 1
            score += 1
            reward = 0.5
        else:
            grid[x][y] = 0
            score -= 1
            reward = -0.5

        print('Score', score)
        is_done = score == (width * height)
        if is_done:
            reward += 5
        return reward, is_done, score
    else:
        return -1, False, score


agent = Agent()

steps = 0

while True:
    state_old = agent.get_state()

    final_move = agent.get_action(state_old)

    reward, done, score = perform_action(final_move)
    steps += 1
    state_new = agent.get_state()
    agent.train_short_memory(state_old, final_move, reward, state_new, done)

    agent.remember(state_old, final_move, reward, state_new, done)
    if (done):
        print("Done!", steps)
        agent.train_long_memory()
        
        steps = 0
        grid = []
        for x in range(width):
            grid.append([0] * height)
        


