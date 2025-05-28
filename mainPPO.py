import sys, random, pickle
import numpy as np
from collections import deque
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from trackObservations import getClosestCenterlinePoint, getCurrentRoadBlock, simulate_lidar_raycast
import utils
from ppo import PPO, Memory
import torch

NUM_BEAMS = 12
STATE_STACK = 3

class RewardNormalizer:
    def __init__(self, eps=1e-8):
        self.rewards = deque(maxlen=10000)
        self.eps = eps

    def normalize(self, reward):
        self.rewards.append(reward)
        mean = np.mean(self.rewards)
        std = np.std(self.rewards) + self.eps
        return (reward - mean) / std

reward_norm = RewardNormalizer()
ppo_agent = PPO(input_dim=1 + NUM_BEAMS * STATE_STACK, action_dim=6)
memory = Memory()
update_interval = 128

class MainClient(Client):
    def __init__(self):
        super().__init__()
        with open("states/trainingStates.sim", "rb") as f:
            self.states = pickle.load(f)

        self.prevDistance = 0.0
        self.expObject = None
        self.lidar_history = deque(maxlen=STATE_STACK)
        self.warming = STATE_STACK

    def on_registered(self, iface: TMInterface):
        print(f"Connected to {iface.server_name}")

    def get_reward(self, position, block):
        curr = getClosestCenterlinePoint(position, block)
        reward = curr - self.prevDistance
        self.prevDistance = curr
        return reward_norm.normalize(reward)

    def process_terminal(self, iface, state, action, reward, logp, value):
        memory.store(state, action, logp, reward, True, value)

        if len(memory.states) >= update_interval:
            with torch.no_grad():
                last_value = torch.tensor(0.0).cuda()
            memory.returns = ppo_agent.compute_gae(memory.rewards, memory.dones, memory.values, last_value.item())
            ppo_agent.update(memory)
            print(f"[Update] Steps: {len(memory.states)}, Avg Reward: {np.mean(memory.rewards):.4f}")
            memory.clear()

        new_state = random.choice(self.states)
        new_state.position[0] += random.uniform(-1, 1)
        new_state.position[2] += random.uniform(-1, 1)
        iface.rewind_to_state(new_state)
        self.prevDistance = getClosestCenterlinePoint(new_state.position, getCurrentRoadBlock(new_state.position))
        self.expObject = None
        self.lidar_history.clear()
        self.warming = STATE_STACK

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time > 0 and _time % 80 == 0:
            tmi_state = iface.get_simulation_state()
            pos = tmi_state.position
            block = getCurrentRoadBlock(pos)

            if block is None or abs(tmi_state.yaw_pitch_roll[2]) > 0.15:
                if self.expObject:
                    self.process_terminal(iface, *self.expObject, -10.0)
                return

            lidar = simulate_lidar_raycast(tmi_state, block, NUM_BEAMS)
            lidar = np.array(lidar) / 60.0
            self.lidar_history.append(lidar)

            if self.warming > 0:
                self.warming -= 1
                return

            stacked = np.concatenate(self.lidar_history)
            speed = tmi_state.display_speed * (-1 if tmi_state.scene_mobil.engine.rear_gear else 1)
            speed /= 60.0
            agent_input = torch.tensor(np.concatenate(([speed], stacked)), dtype=torch.float32).cuda()

            reward = self.get_reward(pos, block)

            if self.expObject:
                state, action, logp, value = self.expObject
                memory.store(state, action, logp, reward, False, value)

            if len(memory.states) >= update_interval:
                with torch.no_grad():
                    last_value = ppo_agent.model(agent_input.unsqueeze(0))[1].item()
                memory.returns = ppo_agent.compute_gae(memory.rewards, memory.dones, [v.item() for v in memory.values], last_value)
                ppo_agent.update(memory)
                print(f"[Update] Steps: {len(memory.states)}, Avg Reward: {np.mean(memory.rewards):.4f}")
                memory.clear()

            with torch.no_grad():
                action, logp, value = ppo_agent.act(agent_input.unsqueeze(0))

            self.expObject = (agent_input, action, logp, value)
            utils.play_action(iface, action.item())

        elif _time > 0 and self.expObject:
            tmi_state = iface.get_simulation_state()
            pos = tmi_state.position
            block = getCurrentRoadBlock(pos)
            if block is None or abs(tmi_state.yaw_pitch_roll[2]) > 0.15:
                self.process_terminal(iface, *self.expObject, -10.0)

if __name__ == "__main__":
    server = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    print(f"Connecting to {server}...")
    try:
        run_client(MainClient(), server)
    finally:
        print("Client shut down.")
