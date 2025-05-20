import torch
import random
import numpy as np
import pickle

from collections import deque
from model import *
from copy import deepcopy
from utils import plot_data

MAX_MEMORY = 100_000
BATCH_SIZE = 64
MIN_EPSILON = 0.12
ACTION_SPACE = 6
epsilon_decay = 0.999
epsilon = 1
n_games = 0

#plot_data(loss_plot)
def save_memories():
    with open('./states/memory_deque.pkl', 'wb') as file:
        pickle.dump(memory, file)

def load_memories():
    global memory
    with open('./states/memory_deque.pkl', 'rb') as file:
        memory = pickle.load(file)
        memory = deque(memory, maxlen=MAX_MEMORY)  # popleft()
        print("Loaded previous memory of size", len(memory), "elements.")


#load_memories()
memory = deque(maxlen=MAX_MEMORY)
model = Linear_QNet(7, 64, 64, ACTION_SPACE).cuda()
target_model = deepcopy(model)
trainer = QTrainer(model, target_model, lr=0.0005, gamma=0.99, TAU=0.005)

# model.load_state_dict(torch.load("model/model.pth"))
# target_model.load_state_dict(torch.load("model/model.pth"))
# print("Loaded models.")


def get_action(state):
    final_action = [0] * ACTION_SPACE

    if random.uniform(0,1) <= epsilon:
        action = random.randint(0, ACTION_SPACE-1)
        final_action[action] = 1

    else:
        state_tensor = torch.tensor(state, dtype=torch.float).cuda()
        prediction = model(state_tensor)
        action = torch.argmax(prediction).item()
        final_action[action] = 1

    return final_action

def train_short_memory(state, action, reward, next_state, game_over):
    trainer.train_step(state, action, reward, next_state, game_over)

def train_long_memory(): #every time run restarts
    if len(memory) > BATCH_SIZE:
        mini_sample = random.sample(memory, BATCH_SIZE) # list of tuples
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        trainer.train_step(states, actions, rewards, next_states, dones)

def remember(state, action, reward, next_state, game_over):
    memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached