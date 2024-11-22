from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys



roadBlocks = []

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')
        

    def on_run_step(self, iface: TMInterface, _time: int):

        if _time > 0 and _time % 1000 == 0:
            state = iface.get_simulation_state()
            pos = state.position
            for i in range(60):
                for j in range(60):
                    if i*16-8 < pos[0] < i*16+8:
                        if j*16-8 < pos[2] < j*16+8:
                            blockCenter = (i*16, j*16)
                            if blockCenter not in roadBlocks:
                                roadBlocks.append(blockCenter)
                                print(blockCenter)
                                
if __name__ == "__main__":
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)



