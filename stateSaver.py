from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from trackObservations import *
import pickle, sys
import random
from utils import saveFile
import time
states = []


with open(r"noInputStates.sim", "rb") as file:
    states = pickle.load(file)

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()
        self.i = 0

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')
        

    def on_run_step(self, iface: TMInterface, _time: int):

        if _time > 0 and _time % 1000 == 0:
            interface_state = iface.get_simulation_state()

            iface.rewind_to_state(states[0])

            # currentRoadBlockIndex = None
            # while currentRoadBlockIndex is None:
            #     pos = random_state.position
            #     pos[0] += random.randint(-3, 3)
            #     pos[2] += random.randint(-3, 3)
            #     currentRoadBlockIndex = getCurrentRoadBlock(pos)

            # random_state.position = pos
            # states.append(random_state)
            # print(len(states))
            



if __name__ == "__main__":
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)



