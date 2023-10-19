import torch
import random
import numpy as np
import pickle
import sys
from collections import deque
from model import *

MAX_MEMORY = 100_000
BATCH_SIZE = 512
MIN_EPSILON = 0.05
epsilon_decay = 0.01
epsilon = 2 # 1 + warming
n_games = 0

def save_memories():
    with open('memory_deque.pkl', 'wb') as file:
        pickle.dump(memory, file)

def load_memories():
    global memory
    with open('memory_deque.pkl', 'rb') as file:
        memory = pickle.load(file)
        print("Loaded previous memory of size", len(memory), "elements.")


        
memory = deque(maxlen=MAX_MEMORY) # popleft()
#load_memories()
model = Linear_QNet(8, 80, 80, 6)
target_model = Linear_QNet(8, 80, 80, 6)
#model.load_state_dict(torch.load("model/model.pth"))
trainer = QTrainer(model, target_model, lr=0.001, gamma=0.99, target_update_every=5)

def get_action(state):
    final_action = [0,0,0,0,0,0]
    
    if random.uniform(0,1) <= epsilon:
        action = random.randint(0,5)
        final_action[action] = 1

    else:
        state_tensor = torch.tensor(state, dtype=torch.float)
        prediction = model(state_tensor)
        action = torch.argmax(prediction).item()
        final_action[action] = 1

    return final_action

def train_short_memory(state, action, reward, next_state, game_over):
    trainer.train_step(state, action, reward, next_state, game_over)

def train_long_memory(): #every time run restarts
    global epsilon
    epsilon = max(MIN_EPSILON, epsilon - epsilon_decay)
        
    if len(memory) > BATCH_SIZE:
        mini_sample = random.sample(memory, BATCH_SIZE) # list of tuples
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        trainer.train_step(states, actions, rewards, next_states, dones)
    
def remember(state, action, reward, next_state, game_over):
    memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached



