import torch
import random
import numpy as np
import pickle
import sys
import cProfile

from collections import deque
from model import *
from utils import plot_data
from model import plot as p
from multiprocessing import Process


MAX_MEMORY = 50_000
BATCH_SIZE = 512
MIN_EPSILON = 0.1
WARMING = 500
epsilon_decay = 0.001
epsilon = 1 + WARMING*epsilon_decay# 1 + warming
n_games = 0

def save_memories():
    with open('memory_deque.pkl', 'wb') as file:
        pickle.dump(memory, file)

def load_memories():
    global memory
    with open('memory_deque.pkl', 'rb') as file:
        memory = pickle.load(file)
        print("Loaded previous memory of size", len(memory), "elements.")


        
memory = deque(maxlen=MAX_MEMORY)  # popleft()
#load_memories()
model = Linear_QNet(8, 80, 80, 6).cuda()
target_model = Linear_QNet(8, 80, 80, 6).cuda()
trainer = QTrainer(model, target_model, lr=0.001, gamma=0.995, target_update_every=100)

#model.load_state_dict(torch.load("model/record_model.pth"))
#target_model.load_state_dict(torch.load("model/record_model.pth"))


def test_train():
    mini_sample = random.sample(memory, BATCH_SIZE) # list of tuples
    states, actions, rewards, next_states, dones = zip(*mini_sample)
    trainer.train_step(states, actions, rewards, next_states, dones)
    trainer.episodes += 1
    print("Finished training")

def t():
    p = Process(target=test_train)
    p.start()

if __name__ == "__main__":
    cProfile.run("t()", sort="cumulative")


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
    pass

def train_long_memory(): #every time run restarts
    global epsilon
    epsilon = max(MIN_EPSILON, epsilon - epsilon_decay)
        
    if len(memory) > BATCH_SIZE:
        p = Process(target=train_batch)
        p.start()

def train_batch():
    if len(memory) > BATCH_SIZE:
        mini_sample = random.sample(memory, BATCH_SIZE) # list of tuples
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        trainer.train_step(states, actions, rewards, next_states, dones)

def remember(state, action, reward, next_state, game_over):
    memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached



