from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from trackObservations import *

import sys
import threading
import time
import numpy as np
import os
import agent

human = False

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.prevSpeed = 0
        self.prevDistance = 0
        self.expObject = None #expObject = list [state, action it took, reward for action, new state, done]
        self.record = 0

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def get_reward(self, car_position, currentRoadBlockIndex):
        #Calculate reward
        currentDistance = getDistanceFromStart(car_position, currentRoadBlockIndex) + 12.8
        reward = (currentDistance - self.prevDistance)
        self.prevDistance = currentDistance 
        return reward

    def process_episode(self):
        #Reset env
        agent.n_games += 1
        agent.train_long_memory()
        agent.trainer.episodes += 1
        self.expObject = None

        #Update scores
        if self.prevDistance > self.record:
            self.record = self.prevDistance
            agent.target_model.save()

        print('Game', agent.n_games, "Score:", round(self.prevDistance,2), 'Record:', round(self.record,2), "Exploration %:", round(agent.epsilon,2)*100)


    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0 and _time % 50 == 0: #20fps
            interface_state = iface.get_simulation_state()
            car_position = interface_state.position
            currentRoadBlockIndex = getCurrentRoadBlock(car_position)

            if currentRoadBlockIndex is not None:
                agent_state = getAgentInputs(interface_state, currentRoadBlockIndex, self.prevSpeed)
                self.prevSpeed = agent_state[0] #used for calculating accel
                #print(f"Speed: {agent_state[0]:.2f} Accel: {agent_state[1]:.2f}")
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
                        #print(self.expObject)


                    #State action for the current iteration
                    state_old = agent_state
                    final_action = agent.get_action(state_old)
                    self.expObject = [state_old, final_action]
                    play_action(iface, final_action)

            else: #Died - currentRoadBlockIndex is None
                if not human:
                    self.expObject += [-10, [0]*8, True] #punishment, empty state, game is done
                    agent.train_short_memory(*self.expObject)
                    agent.remember(*self.expObject)
                    iface.respawn() #Reset env
                    self.process_episode() #Train long memory 


server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
print(f'Connecting to {server_name}...')
run_client(MainClient(), server_name)



