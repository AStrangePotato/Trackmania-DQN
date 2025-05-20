from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from trackObservations import *

import sys
import numpy as np
import agent
import utils
import pickle
import random
from multiprocessing import shared_memory

human = False

# Create a shared memory buffer
shm = shared_memory.SharedMemory(create=True, name="tmdata", size=80)  # 3 floats for x, y, z + 7 floats for agent state
shared_data = np.ndarray((10,), dtype=np.float64, buffer=shm.buf)  # Total of 10 floats

print("Shared memory created:", shm.name)

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()        
        with open(r"states/trainingStates.sim", "rb") as file:
            self.states = pickle.load(file)
        self.prevSpeed = 0
        self.prevDistance = 0
        self.expObject = None #expObject = list [state, action it took, reward for action, new state, done]
        self.record = 0

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def get_reward(self, car_position, currentRoadBlockIndex):
        currentDistance = getClosestCenterlinePoint(car_position, currentRoadBlockIndex)
        reward = (currentDistance - self.prevDistance)
        self.prevDistance = currentDistance 
        return reward * 5

    def process_episode(self):
        self.expObject += [-5, [0] * 7, True] #reward, empty state of length input space, game is done

        agent.train_short_memory(*self.expObject)
        agent.remember(*self.expObject)

        agent.n_games += 1
        agent.epsilon = max(agent.MIN_EPSILON, agent.epsilon * agent.epsilon_decay)

        agent.trainer.episodes += 1
        self.expObject = None

        agent.train_long_memory()

        print('Game', agent.n_games, "Exploration %:", round(agent.epsilon,3)*100)


    def on_run_step(self, iface: TMInterface, _time: int):
        if _time > 0 and _time % 120 == 0: #10fps, 1000ms/s
            interface_state = iface.get_simulation_state()

            car_position = interface_state.position
            currentRoadBlockIndex = getCurrentRoadBlock(car_position)

            if currentRoadBlockIndex is not None and abs(interface_state.yaw_pitch_roll[2]) < 0.1:
                agent_state = getAgentInputs(interface_state, currentRoadBlockIndex, self.prevSpeed)

                self.prevSpeed = agent_state[0] #used for calculating accel
                reward = -agent_state[3] + agent_state[0] * 15
                
                print(reward)
                #Load into shared memory for visualization
                shared_data[:3] = car_position  # First 3 values for car position [x, y, z]
                shared_data[3:] = agent_state   # Next 7 values for agent state [a, b, c, d, e, f, g]

                #Calculate experience object based off last iteration's state action
                if not human:
                    if self.expObject is not None:
                        self.expObject += [reward, agent_state, False]
                        
                        #Train short memory
                        agent.train_short_memory(*self.expObject)
                        agent.remember(*self.expObject)

                    #State action for the current iteration
                    state_old = agent_state
                    final_action = agent.get_action(state_old)
                    self.expObject = [state_old, final_action]
                    utils.play_action(iface, final_action)

            else: #Died - currentRoadBlockIndex is None
                if not human:
                    if self.expObject is not None:
                        random_state = random.choice(self.states)
                        random_state.position[0] += random.uniform(-1, 1)  # Add random offset to x
                        random_state.position[2] += random.uniform(-1, 1)  # Add random offset to z
                        iface.rewind_to_state(random_state)
                        self.process_episode() #Train long memory 



if __name__ == "__main__":
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Attempting to connect to {server_name}...')
    run_client(MainClient(), server_name)



def sv():
    agent.save_memories()
    agent.target_model.save()
    print("Saved shit.")