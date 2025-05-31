import sys, random, pickle
import numpy as np
import torch
import os
from collections import deque
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from trackObservations import *
import utils
from ppo import PPO, Memory

NUM_BEAMS = 12
STATE_STACK = 3
UPDATE_INTERVAL = 2048

class MainClient(Client):
    def __init__(self):
        super().__init__()
        self.ppo_agent = PPO(input_dim=3 + NUM_BEAMS * STATE_STACK, action_dim=8)
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

    def on_registered(self, iface: TMInterface):
        print(f"Connected to {iface.server_name}")
        new_state = iface.get_simulation_state()
        self.prevDistance = getClosestCenterlinePoint(new_state.position, getCurrentRoadBlock(new_state.position))
        iface.set_timeout(5000)

    def get_reward(self,position, block):
        curr = getClosestCenterlinePoint(position, block)
        reward = curr - self.prevDistance
        self.prevDistance = curr
        return reward

    def reset_episode(self, iface, terminal_reward):
        if self.last_state is not None:
            self.memory.store(self.last_state, self.last_action, self.last_logp, terminal_reward, True, self.last_value)

        new_state = random.choice(self.states)
        iface.rewind_to_state(new_state)

        self.prevDistance = getClosestCenterlinePoint(new_state.position, getCurrentRoadBlock(new_state.position))
        self.lidar_history.clear()
        self.warming = STATE_STACK
        self.last_state = self.last_action = self.last_logp = self.last_value = None

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time > 0 and _time % 80 == 0:
            tmi_state = iface.get_simulation_state()
            pos = tmi_state.position
            block = getCurrentRoadBlock(pos, self.rb_guess)

            if block is None:
                self.reset_episode(iface, -100.0)
                return

            self.rb_guess = block

            lidar = np.array(simulate_lidar_raycast(tmi_state, block, NUM_BEAMS)) / 60.0
            self.lidar_history.append(lidar)

            if self.warming > 0:
                self.warming -= 1
                iface.set_input_state(**{'left': False, 'right': False, 'accelerate': False, 'brake': False})
                return

            stacked = np.concatenate(self.lidar_history)
            speed = tmi_state.display_speed * (-1 if tmi_state.scene_mobil.engine.rear_gear else 1) / 60.0
            turning_rate = tmi_state.scene_mobil.turning_rate

            next_turn = getNextTurnDirection(block)
            state_tensor = torch.tensor(np.concatenate(([speed, turning_rate, next_turn], stacked)), dtype=torch.float32).cuda()

            reward = self.get_reward(pos, block) * 10
            reward += speed - 0.5

            if self.last_state is not None:
                self.memory.store(self.last_state, self.last_action, self.last_logp, reward, False, self.last_value)
            
            #finish race
            if block >= 918:
                self.reset_episode(iface, int(self.ppo_agent.get_value(state_tensor).cpu().item()))

            if len(self.memory.states) >= UPDATE_INTERVAL:
                with torch.no_grad():
                    last_value = self.ppo_agent.get_value(state_tensor).item()
                self.memory.returns = self.ppo_agent.compute_gae(self.memory.rewards, self.memory.dones, self.memory.values, last_value)
                self.ppo_agent.update(self.memory)
                print(f"[Update] Total Steps: {self.ppo_agent.global_step}, Avg Reward: {np.mean(self.memory.rewards):.4f}")
                # Save checkpoint every 10 updates
                if self.ppo_agent.global_step % (UPDATE_INTERVAL * 10) == 0:
                    self.ppo_agent.save_checkpoint()
                self.memory.clear()

            with torch.no_grad():
                action, logp, value = self.ppo_agent.act(state_tensor.unsqueeze(0))

            self.last_state = state_tensor
            self.last_action = action
            self.last_logp = logp
            self.last_value = value
            utils.play_action(iface, action.item())

def main():
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

if __name__ == "__main__":
    main()