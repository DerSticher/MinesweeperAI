import torch
import random
import numpy as np
from collections import deque
from Minesweeper import Minesweeper
from Model import Linear_QNet, QTrainer

MAX_MEMORY = 1_000_000
BATCH_SIZE = 10_000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self, game):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(game.width * game.height, game.width * game.height * 4, 3)
        self.trainer = QTrainer(self.model, LEARNING_RATE, self.gamma)
        self.use_cuda = False # torch.cuda.is_available()

        if (self.use_cuda):
            self.model.to(device='cuda')

        self.game = game

    def get_state(self):
        state = []
        for x in range(self.game.width):
            for y in range(self.game.height):
                tile = self.game.get_tile((x, y))
                if tile.is_shown:
                    state.append(tile.number_of_close_bombs)
                elif tile.is_marked:
                    state.append(-1)
                else:
                    state.append(-2)
                    
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
        self.epsilon = 80 * pow(0.99, self.n_games)
        
        if (random.randint(0, 200) < self.epsilon):
            action = random.randint(0, 2) % 2
            x = random.randint(-3, self.game.width + 3)
            y = random.randint(-3, self.game.height + 3)
            prediction = [action, x, y]
            print('Random Action', prediction)
            return prediction
        else:
            if (self.use_cuda):
                state0 = torch.tensor(state, dtype=torch.float).cuda()
            else:
                state0 = torch.tensor(state, dtype=torch.float)

            prediction = self.model(state0)
            # print('"Smart" Action', prediction)
            # TODO: idk 

            action = round(float(prediction[0]))
            x = round(float(prediction[1]) * self.game.width)
            y = round(float(prediction[2]) * self.game.height)

            if (action == 0 or action == 1) and 0 <= x < self.game.width and 0 <= y < self.game.height:
                print('Smart Prediction!', action, x, y)
            else:
                print('Dumb Prediction', action, x, y)


            return [
                int(prediction[0]), 
                int(prediction[1]), 
                int(prediction[2])]







