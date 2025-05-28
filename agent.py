import torch
import random
import numpy as np
import pickle
from collections import deque
from model import *

MAX_MEMORY = 150_000
BATCH_SIZE = 64
MIN_EPSILON = 0.10
INPUT_SPACE = 1 + 3*12
ACTION_SPACE = 8
WARMING = 5000
epsilon_decay = 0.998
epsilon = 0.6
n_games = 0


def save_memories():
    with open('./states/memory_deque.pkl', 'wb') as file:
        pickle.dump(memory, file)

def load_memories():
    global memory
    with open('./states/memory_deque.pkl', 'rb') as file:
        memory = pickle.load(file)
        memory = deque(memory, maxlen=MAX_MEMORY)  # popleft()
        print("Loaded previous memory of size", len(memory), "elements.")


load_memories()
#memory = deque(maxlen=MAX_MEMORY)
model = Linear_QNet(INPUT_SPACE, 128, 64, ACTION_SPACE).cuda()
target_model = Linear_QNet(INPUT_SPACE, 128, 64, ACTION_SPACE).cuda()
trainer = QTrainer(model, target_model, lr=3e-5, gamma=0.99, tau=0.001)

model.load_state_dict(torch.load("model/model.pth"))
target_model.load_state_dict(torch.load("model/model.pth"))
print("Loaded models.")

total_steps = 32512

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

import numpy as np

TARGET_SPEED = 25
MAX_STEER = 65536
MAX_ANGLE_RAD = np.deg2rad(60.0)
beam_angles_deg = np.array([
    -60.0, -40.0, -20.0, -10.0, -5.0, -2.0,
     2.0,   5.0,  10.0,  20.0, 40.0, 60.0
])
beam_angles_rad = np.deg2rad(beam_angles_deg)

def get_heuristic_action(state):
    speed = state[0] * 60
    lidar_raw = np.array(state[1:13])

    # Avoid zero division
    weights = lidar_raw / (np.sum(lidar_raw) + 1e-8)

    # Weighted mean direction
    mean_angle_rad = np.sum(weights * beam_angles_rad)

    # Normalize [-1, 1] based on max angle range
    steer_norm = mean_angle_rad / MAX_ANGLE_RAD

    steer = int(np.clip(steer_norm * MAX_STEER * -1 * 15, -MAX_STEER, MAX_STEER))

    accelerate = speed < TARGET_SPEED - 15
    brake = speed > TARGET_SPEED

    return {
        "steer": steer,
        "accelerate": accelerate,
        "brake": brake
    }


def train_short_memory(state, action, reward, next_state, game_over):
    if total_steps < WARMING:
        return
    trainer.train_step(state, action, reward, next_state, game_over)

def train_long_memory(): #every time run restarts
    if total_steps < WARMING:
        return
    if len(memory) > BATCH_SIZE:
        mini_sample = random.sample(memory, BATCH_SIZE) # list of tuples
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        trainer.train_step(states, actions, rewards, next_states, dones)

def remember(state, action, reward, next_state, game_over):
    global total_steps
    total_steps += 1
    memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached