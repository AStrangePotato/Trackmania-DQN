import torch
import random
import numpy as np
import pickle

from collections import deque
from model import *
from utils import plot_data
from model import loss_plot as p
from copy import deepcopy

MAX_MEMORY = 200_000
BATCH_SIZE = 128
MIN_EPSILON = 0.2
ACTION_SPACE = 6
epsilon_decay = 0.0002
epsilon = 0.75
temperature = 1
n_games = 0


def save_memories():
    with open('/states/memory_deque.pkl', 'wb') as file:
        pickle.dump(memory, file)

def load_memories():
    global memory
    with open('memory_deque.pkl', 'rb') as file:
        memory = pickle.load(file)
        memory = deque(memory, maxlen=MAX_MEMORY)  # popleft()
        print("Loaded previous memory of size", len(memory), "elements.")


#load_memories()
memory = deque(maxlen=MAX_MEMORY)
model = Linear_QNet(7, 256, 128, ACTION_SPACE).cuda()
target_model = deepcopy(model)
trainer = QTrainer(model, target_model, lr=0.0003, gamma=0.99, TAU=0.005)

#model.load_state_dict(torch.load("model/record_model.pth"))
#target_model.load_state_dict(torch.load("record_model/model.pth"))
#print("Loaded models.")


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

def get_action_boltzmann(state):
    final_action = [0] * ACTION_SPACE

    # Convert state to tensor and predict Q-values
    state_tensor = torch.tensor(state, dtype=torch.float).cuda()
    q_values = model(state_tensor).detach().cpu().numpy()

    # Compute softmax probabilities
    exp_q = np.exp(q_values / temperature)
    probabilities = exp_q / np.sum(exp_q)

    # Select an action based on probabilities
    action = np.random.choice(len(q_values), p=probabilities)
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