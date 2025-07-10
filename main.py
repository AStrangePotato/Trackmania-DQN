import sys, random, pickle
import numpy as np
import torch
import os
from collections import deque
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from trackObservations import *
from ppo import PPO, Memory
import utils

NUM_BEAMS = 13
STATE_STACK = 3
UPDATE_INTERVAL = 8192

class MainClient(Client):
    def __init__(self):
        super().__init__()
        self.ppo_agent = PPO(input_dim=5 + NUM_BEAMS * STATE_STACK, action_dim=len(utils.action_space))
        self.memory = Memory()
        with open("states/states.sim", "rb") as f:
            self.states = pickle.load(f)

        self.prevDistance = 0.0
        self.lidar_history = deque(maxlen=STATE_STACK)
        self.warming = STATE_STACK
        self.last_state = None
        self.last_action = None
        self.last_logp = None
        self.last_value = None
        self.rb_guess = 0

        self.finished = False

    def on_registered(self, iface: TMInterface):
        print(f"Connected to {iface.server_name}")
        new_state = random.choice(self.states)
        iface.rewind_to_state(new_state)
        self.prevDistance = getClosestCenterlinePoint(new_state.position)
        iface.set_timeout(15000)

    def get_reward(self, position):
        curr = getClosestCenterlinePoint(position)
        reward = curr - self.prevDistance
        self.prevDistance = curr
        return reward

    def reset_episode(self, iface, terminal_reward):
        if self.last_state is not None:
            self.memory.store(self.last_state, self.last_action, self.last_logp, terminal_reward, True, self.last_value)

        new_state = random.choice(self.states)
        iface.rewind_to_state(new_state)

        self.prevDistance = getClosestCenterlinePoint(new_state.position)
        self.lidar_history.clear()
        self.warming = STATE_STACK
        self.last_state = self.last_action = self.last_logp = self.last_value = None

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == 1:
            self.finished = True
            iface.prevent_simulation_finish()
            
            
    def on_run_step(self, iface: TMInterface, _time: int):
        if _time > 0 and _time % 50 == 0:
            tmi_state = iface.get_simulation_state()
            pos = tmi_state.position
            block = getCurrentRoadBlock(pos, self.rb_guess)

            if self.finished:
                self.finished = False
                self.reset_episode(iface, 100) #terminal reward?

            if block is None or abs(tmi_state.yaw_pitch_roll[2]) > 0.1 or abs(tmi_state.yaw_pitch_roll[1]) > 0.1:
                self.reset_episode(iface, -10)
                return

            self.rb_guess = block

            lidar = np.array(simulate_lidar_raycast(tmi_state, block, max_range=100)) / 80.0
            self.lidar_history.append(lidar)

            if self.warming > 0:
                self.warming -= 1
                iface.set_input_state(**{'left': False, 'right': False, 'accelerate': False, 'brake': False})
                return

            stacked = np.concatenate(self.lidar_history)
            speed = tmi_state.display_speed * (-1 if tmi_state.scene_mobil.engine.rear_gear else 1) / 120.0
            turning_rate = tmi_state.scene_mobil.turning_rate
            next_turn = getNextTurnDirection(block)
            next_next_turn = getNextTurnDirection(getCenterlineEndblock(block, retIndex=True))
            dist_to_next = (getDistanceToNextTurn(tmi_state, block) - 8) / 160 - 1
            state_tensor = torch.tensor(np.concatenate(([speed, turning_rate, next_turn, next_next_turn, dist_to_next], stacked)), dtype=torch.float32).cuda()

            reward = self.get_reward(pos)
            reward += speed/10

            if self.last_state is not None:
                self.memory.store(self.last_state, self.last_action, self.last_logp, reward, False, self.last_value)
            
            if len(self.memory.states) >= UPDATE_INTERVAL:
                with torch.no_grad():
                    last_value = self.ppo_agent.get_value(state_tensor).item()
                self.memory.returns = self.ppo_agent.compute_gae(self.memory.rewards, self.memory.dones, self.memory.values, last_value)
                self.ppo_agent.update(self.memory)
                print(f"[Update] Total Steps: {self.ppo_agent.global_step}, Avg Reward: {np.mean(self.memory.rewards):.4f}")

                if self.ppo_agent.global_step % (UPDATE_INTERVAL * 10) == 0 or random.randint(0, 10) == 0:
                    self.ppo_agent.save_checkpoint()

                self.memory.clear()

            with torch.no_grad():
                action, logp, value = self.ppo_agent.act(state_tensor.unsqueeze(0))

            self.last_state = state_tensor
            self.last_action = action
            self.last_logp = logp
            self.last_value = value
            iface.set_input_state(**utils.action_space[action.item()])

if __name__ == "__main__":
    server = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    print(f"Connecting to {server}...")

    # Load latest checkpoint if available
    checkpoint_dir = "checkpoints"
    client = MainClient()
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            client.ppo_agent.load_checkpoint(checkpoint_path)

    run_client(client, server)