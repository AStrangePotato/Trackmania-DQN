import sys
import os
import numpy as np
import torch
import pickle
from collections import deque
from multiprocessing import shared_memory
import atexit
from ppo import PPO
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from archived.trackObservations import *
import utils

NUM_BEAMS = 12
STATE_STACK = 3
SAVE_INTERVAL = 400

states, cp, saved_snapshots = [], [], []

class EvalClient(Client):
    def __init__(self, shm):
        super().__init__()
        self.ppo_agent = PPO(input_dim=3 + NUM_BEAMS * STATE_STACK, action_dim=8)
        self.lidar_history = deque(maxlen=STATE_STACK)
        self.prevDistance = 0.0
        self.rb_guess = 0
        self.shared_data = np.ndarray((NUM_BEAMS,), dtype=np.float64, buffer=shm.buf)
        self.ticks = 0

    def on_registered(self, iface: TMInterface):
        print(f"Connected to {iface.server_name}")

    def save_snapshot(self, tmi_state):
        # Save deep copies of everything
        saved_snapshots.append((tmi_state, list(cp), list(states)))
        print(f"Snapshot saved at tick {self.ticks}, total: {len(saved_snapshots)}")

    def restore_latest_snapshot(self, iface):
        if saved_snapshots:
            last_state, last_cp, last_states = saved_snapshots[-1]
            iface.rewind_to_state(last_state)
            cp.clear()
            cp.extend(last_cp)
            states.clear()
            states.extend(last_states)
            self.lidar_history.clear()
            print("Restored latest snapshot.")
        else:
            iface.respawn()
            cp.clear()
            states.clear()
            self.lidar_history.clear()
            print("No snapshot. Full respawn.")

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time % 80 == 0:
            self.ticks += 1

            tmi_state = iface.get_simulation_state()
            pos = tmi_state.position
            block = getCurrentRoadBlock(pos, self.rb_guess)



            if block is None or abs(tmi_state.yaw_pitch_roll[2]) > 0.15:
                print("Agent crashed or off track.")
                self.restore_latest_snapshot(iface)
                return
            
            if block > 920:
                sys.exit()
                
            if self.ticks % SAVE_INTERVAL == 0:
                self.save_snapshot(tmi_state)

            if block != self.rb_guess:
                if block % 2 == 0:
                    states.append(tmi_state)
                cp.append((pos[0], pos[2]))

            self.rb_guess = block
            lidar_scan = np.array(simulate_lidar_raycast(tmi_state, block, NUM_BEAMS)) / 60.0
            self.lidar_history.append(lidar_scan.tolist())
            self.shared_data[:] = lidar_scan

            if len(self.lidar_history) < STATE_STACK:
                iface.set_input_state(**{'left': False, 'right': False, 'accelerate': False})
                return

            stacked = np.concatenate(self.lidar_history)
            speed = tmi_state.display_speed * (-1 if tmi_state.scene_mobil.engine.rear_gear else 1) / 60.0
            turning_rate = tmi_state.scene_mobil.turning_rate
            next_turn = getNextTurnDirection(block)
            state_tensor = torch.tensor(np.concatenate(([speed, turning_rate, next_turn], stacked)), dtype=torch.float32).cuda()

            with torch.no_grad():
                action, _, _ = self.ppo_agent.act(state_tensor.unsqueeze(0))

            utils.play_action(iface, action.item())


def main():
    server = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    print(f"Connecting to {server}...")

    shm = shared_memory.SharedMemory(create=True, size=NUM_BEAMS * 8, name='tmdata')
    atexit.register(lambda: shm.close())

    client = EvalClient(shm)

    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            client.ppo_agent.load_checkpoint(os.path.join(checkpoint_dir, latest))

    try:
        run_client(client, server)
    finally:
        shm.unlink()

def sv(file_path="states/states.sim"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(states, f)
    print(f"States saved to {file_path}")

def svcp(file_path="states/rewards.sim"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(cp, f)
    print(f"Gates saved to {file_path}")

if __name__ == "__main__":
    main()
