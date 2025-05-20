from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from trackObservations import *
import sys, os, numpy as np, pickle, random
from multiprocessing import shared_memory
import agent, utils

# shared memory setup (unchanged)
shm = shared_memory.SharedMemory(create=True, name="tmdata", size=80)
shared_data = np.ndarray((10,), dtype=np.float64, buffer=shm.buf)
print("Shared memory created:", shm.name)

class MainClient(Client):
    def __init__(self):
        super().__init__()
        with open("states/trainingStates.sim", "rb") as f:
            self.states = pickle.load(f)

        # placeholders for tracking previous
        self.prevSpeed = 0.0
        self.prevDistance = 0.0

        # holds (state, action) of last step; full transitions built on the fly
        self.expObject = None

        self.human = False

    def on_registered(self, iface: TMInterface):
        print(f"Registered to {iface.server_name}")

    def get_reward(self, car_position, currentRoadBlockIndex):
        curr = getClosestCenterlinePoint(car_position, currentRoadBlockIndex)
        r = curr - self.prevDistance
        self.prevDistance = curr
        return r

    def process_terminal(self, iface, last_state, last_action):
        transition = (last_state, last_action, -10, last_state, True)
        agent.train_short_memory(*transition)
        agent.remember(*transition)

        # complete the episode
        agent.train_long_memory()
        agent.n_games += 1
        agent.epsilon = max(agent.MIN_EPSILON, agent.epsilon * agent.epsilon_decay)
        agent.trainer.episodes += 1
        print(f"Game {agent.n_games:3d}  Exploration: {agent.epsilon*100:.1f}%")

        # pick a random spawn state and reset sim
        random_state = random.choice(self.states)
        # add small noise
        random_state.position[0] += random.uniform(-1,1)
        random_state.position[2] += random.uniform(-1,1)
        iface.rewind_to_state(random_state)

        # reset prevDistance and prevSpeed to avoid spurious first reward
        self.prevDistance = getClosestCenterlinePoint(random_state.position, getCurrentRoadBlock(random_state.position))
        self.prevSpeed = 0.0

        # clear the expObject so next step starts fresh
        self.expObject = None

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time > 0 and _time % 120 == 0:
            tmi_state = iface.get_simulation_state()
            pos = tmi_state.position
            currentRoadBlockIndex = getCurrentRoadBlock(pos)


            if currentRoadBlockIndex is not None and abs(tmi_state.yaw_pitch_roll[2]) < 0.1:
                agent_state = getAgentInputs(tmi_state, currentRoadBlockIndex, self.prevSpeed)
                self.prevSpeed = agent_state[0]

                # compute and clip reward
                raw_r = self.get_reward(pos, currentRoadBlockIndex)
                if tmi_state.display_speed < 15:
                    raw_r -= 30
                reward = raw_r / 30

                print(reward)

                # write to shared memory for visual
                shared_data[:3] = pos
                shared_data[3:] = agent_state

                # if we have a previous (s,a), train on that transition
                if not self.human:
                    if self.expObject is not None:
                        s_old, a_old = self.expObject
                        transition = (s_old, a_old, reward, agent_state, False)
                        agent.train_short_memory(*transition)
                        agent.remember(*transition)
                    
                    # choose and execute new action
                    action = agent.get_action(agent_state)
                    self.expObject = (agent_state, action)
                    utils.play_action(iface, action)

            else: # died / off-track
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