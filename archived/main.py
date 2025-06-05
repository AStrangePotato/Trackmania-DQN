from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from archived.trackObservations import *
import sys, random, pickle
import numpy as np
from multiprocessing import shared_memory
import agent, utils
from collections import deque

NUM_BEAMS = 12

shm = shared_memory.SharedMemory(create=True, name="tmdata", size=NUM_BEAMS * 8) 
shared_data = np.ndarray((NUM_BEAMS,), dtype=np.float64, buffer=shm.buf)
print("Shared memory created:", shm.name)


class MainClient(Client):
    def __init__(self):
        super().__init__()
        with open("states/trainingStates.sim", "rb") as f:
            self.states = pickle.load(f)

        self.prevDistance = 0.0
        self.expObject = None
        self.human = True
        self.warming = 3  # wait for 3 steps before using stacked lidar
        self.lidar_history = deque(maxlen=3)

    def on_registered(self, iface: TMInterface):
        print(f"Registered to {iface.server_name}")
        random_state = self.states[-1]
        random_state.position[0] = 900
        random_state.position[2] = 352
        iface.rewind_to_state(random_state)

    def get_reward(self, car_position, currentRoadBlockIndex):
        curr = getClosestCenterlinePoint(car_position, currentRoadBlockIndex)
        reward = curr - self.prevDistance
        self.prevDistance = curr
        return reward

    def process_terminal(self, iface, last_state, last_action):
        transition = (last_state, last_action, -1, last_state, True)
        agent.train_short_memory(*transition)
        agent.remember(*transition)

        agent.train_long_memory()
        agent.n_games += 1
        agent.epsilon = max(agent.MIN_EPSILON, agent.epsilon * agent.epsilon_decay)
        agent.trainer.episodes += 1
        print(f"Game {agent.n_games:3d}  Exploration: {agent.epsilon*100:.1f}%  Steps: {agent.total_steps}")

        random_state = random.choice(self.states)
        random_state.position[0] = 176
        random_state.position[2] = 800
        iface.rewind_to_state(random_state)

        self.prevDistance = getClosestCenterlinePoint(random_state.position, getCurrentRoadBlock(random_state.position))
        self.expObject = None
        self.warming = 3
        self.lidar_history.clear()

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time > 0 and _time % 80 == 0:
            tmi_state = iface.get_simulation_state()
            pos = tmi_state.position
            currentRoadBlockIndex = getCurrentRoadBlock(pos)

            if currentRoadBlockIndex is not None and abs(tmi_state.yaw_pitch_roll[2]) < 0.1:
                lidar_scan = simulate_lidar_raycast(tmi_state, currentRoadBlockIndex, num_beams=NUM_BEAMS)
                lidar_scan = np.array([x / 60 for x in lidar_scan])

                # Update shared memory with current lidar scan (normalized)
                shared_data[:] = lidar_scan

                self.lidar_history.append(lidar_scan.tolist())

                self.lidar_history.append(lidar_scan)

                if self.warming > 0:
                    self.warming -= 1
                    return  # skip action until we have 3 scans

                # By now, self.lidar_history has exactly 3 scans
                stacked_lidar = np.array(self.lidar_history).flatten().tolist()

                speed = tmi_state.display_speed * (-1 if tmi_state.scene_mobil.engine.rear_gear == 1 else 1)
                agent_state = [speed / 60] + stacked_lidar

                reward = self.get_reward(pos, currentRoadBlockIndex)

                if not self.human:
                    if self.expObject is not None:
                        s_old, a_old = self.expObject
                        transition = (s_old, a_old, reward, agent_state, False)
                        agent.train_short_memory(*transition)
                        agent.remember(*transition)

                    action = agent.get_action(agent_state)
                    self.expObject = (agent_state, action)
                    utils.play_action(iface, action)
                    #iface.set_input_state(**action)

            else:
                if self.expObject is not None and not self.human:
                    last_state, last_action = self.expObject
                    self.process_terminal(iface, last_state, last_action)


if __name__ == "__main__":
    server = f"TMInterface{sys.argv[1]}" if len(sys.argv) > 1 else "TMInterface0"
    print(f"Connecting to {server!r} …")
    run_client(MainClient(), server)


def save():
    agent.save_memories()
    agent.target_model.save()

def plot():
    import matplotlib.pyplot as plt
    plt.plot(agent.loss_plot)
    plt.show()
