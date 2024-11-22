from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from trackObservations import *

import sys
import numpy as np
import agent
import utils
import pickle
import random

human = False

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
        return reward

    def process_episode(self, finish=False):
        if finish:
            self.expObject += [10, [0] * 7, True] #get the ice cream, empty state, game is done
        else:
            self.expObject += [0, [0] * 7, True] #get the whip, empty state, game is done

        agent.train_short_memory(*self.expObject)
        agent.remember(*self.expObject)

        agent.n_games += 1
        agent.epsilon = max(agent.MIN_EPSILON, agent.epsilon - agent.epsilon_decay)
        agent.trainer.episodes += 1
        self.expObject = None

        agent.train_long_memory()

        print('Game', agent.n_games, "Exploration %:", round(agent.epsilon,3)*100)


    def on_run_step(self, iface: TMInterface, _time: int):
        if _time > 0 and _time % 120 == 0: #8fps, 1000ms/s
            interface_state = iface.get_simulation_state()

            car_position = interface_state.position
            currentRoadBlockIndex = getCurrentRoadBlock(car_position)
            ypr = interface_state.yaw_pitch_roll

            if currentRoadBlockIndex is not None and currentRoadBlockIndex < 143 and abs(ypr[2]) < 0.1 and abs(ypr[1]) < 0.05:
                agent_state = getAgentInputs(interface_state, currentRoadBlockIndex, self.prevSpeed)
                self.prevSpeed = agent_state[0] #used for calculating accel
                reward = self.get_reward(car_position, currentRoadBlockIndex)

                #Calculate experience object based off last iteration's state action
                if not human:
                    if self.expObject is not None:
                        self.expObject.append(reward) #reward
                        self.expObject.append(agent_state) #new state
                        self.expObject.append(False) #not game over since currentRoadBlock is valid

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
                        iface.rewind_to_state(random.choice(self.states))
                        if currentRoadBlockIndex is not None and currentRoadBlockIndex >= 143:
                            self.process_episode(finish=True) #Train long memory
                        else:
                            self.process_episode() #Train long memory 



if __name__ == "__main__":
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Attempting to connect to {server_name}...')
    run_client(MainClient(), server_name)



