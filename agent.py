import torch
import random
import numpy as np
import pickle

from collections import deque
from model import *
from utils import plot_data
from model import loss_plot as p

MAX_MEMORY = 200_000
BATCH_SIZE = 256
MIN_EPSILON = 0.1
epsilon_decay = 0.0001
epsilon = 0.2
n_games = 0

def save_memories():
    with open('memory_deque.pkl', 'wb') as file:
        pickle.dump(memory, file)

def load_memories():
    global memory
    with open('memory_deque.pkl', 'rb') as file:
        memory = pickle.load(file)
        memory = deque(memory, maxlen=MAX_MEMORY)  # popleft()
        print("Loaded previous memory of size", len(memory), "elements.")


load_memories()
model = Linear_QNet(7, 128, 128, 6).cuda()
target_model = Linear_QNet(7, 128, 128, 6).cuda()
trainer = QTrainer(model, target_model, lr=0.0002, gamma=0.99, TAU=0.005)

model.load_state_dict(torch.load("model/model.pth"))
target_model.load_state_dict(torch.load("model/model.pth"))
print("Loaded models.")


def get_action(state):
    final_action = [0,0,0,0,0,0]
    if random.uniform(0,1) <= epsilon:
        action = random.randint(0,5)
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



