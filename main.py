from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from trackObservations import *

import sys
import threading
import time
import numpy as np
import os
import agent
import utils

human = False
distance_plot = []


class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.prevSpeed = 0
        self.prevDistance = 0
        self.expObject = None #expObject = list [state, action it took, reward for action, new state, done]
        self.record = 0

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    #Calculate reward
    def get_reward(self, car_position, currentRoadBlockIndex):
        currentDistance = getClosestCenterlinePoint(car_position, currentRoadBlockIndex)
        reward = (currentDistance - self.prevDistance)
        self.prevDistance = currentDistance 
        return reward * utils.inc

    def process_episode(self):
        #Reset env
        agent.n_games += 1
        if agent.n_games > agent.WARMING:
            agent.train_long_memory()
        agent.trainer.episodes += 1
        self.expObject = None
        distance_plot.append(self.prevDistance)
        #Update scores
        if self.prevDistance > self.record:
            self.record = self.prevDistance
            agent.target_model.save("record_model.pth")

        print('Game', agent.n_games, "Score:", round(self.prevDistance,2), 'Record:', round(self.record,2), "Exploration %:", round(agent.epsilon,3)*100)


    def on_run_step(self, iface: TMInterface, _time: int):
        if _time > 0 and _time % 100 == 0: #10fps
            interface_state = iface.get_simulation_state()
            car_position = interface_state.position
            currentRoadBlockIndex = getCurrentRoadBlock(car_position)

            if currentRoadBlockIndex is not None:
                #Get and scale state
                agent_state = getAgentInputs(interface_state, currentRoadBlockIndex, self.prevSpeed)
                self.prevSpeed = agent_state[0] #used for calculating accel
                scaled_state = scaleAgentInputs(*agent_state)
                
                reward = self.get_reward(car_position, currentRoadBlockIndex)

                #Calculate experience object based off last iteration's state action
                if not human:
                    if self.expObject is not None:
                        self.expObject.append(reward) #reward
                        self.expObject.append(scaled_state) #new state
                        self.expObject.append(False) #not game over since currentRoadBlock is valid

                        #Train short memory
                        agent.train_short_memory(*self.expObject)
                        agent.remember(*self.expObject)

                    #State action for the current iteration
                    state_old = scaled_state
                    final_action = agent.get_action(state_old)
                    self.expObject = [state_old, final_action]
                    play_action(iface, final_action)


            else: #Died - currentRoadBlockIndex is None
                if not human:
                    if self.expObject is not None:
                        self.expObject += [-10, [0] * 8, True] #beat the shit out of it, empty state, game is done
                        agent.train_short_memory(*self.expObject)
                        agent.remember(*self.expObject)
                        respawn(iface) #Reset env
                        self.process_episode() #Train long memory 

            #Agent timeout
            if _time > 111830+60000:
                if not human:
                    respawn(iface)  


def respawn(iface):
    iface.rewind_to_state(utils.initial_state)

if __name__ == "__main__":
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)



