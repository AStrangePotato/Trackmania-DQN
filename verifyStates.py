import pickle
import time
from tminterface.client import Client, run_client
from tminterface.interface import TMInterface

# Load states from file
try:
    with open("states/states.sim", "rb") as f:
        replay_states = pickle.load(f)
    print(f"Loaded {len(replay_states)} states from states/states.sim")
except Exception as e:
    print(f"Error loading states: {e}")
    replay_states = []

class ReplayClient(Client):
    def __init__(self):
        super().__init__()
        self.index = 0
        self.last_step_time = time.time()

    def on_registered(self, iface: TMInterface):
        print(f"Connected to {iface.server_name}")
        iface.set_timeout(5000)
        if replay_states:
            iface.rewind_to_state(replay_states[0])
        else:
            print("No states to replay.")

    def on_run_step(self, iface: TMInterface, _time: int):
        if not replay_states:
            return

        now = time.time()
        if now - self.last_step_time > 0.1:  # Step every 0.5 seconds
            self.index += 1
            self.last_step_time = now
            if self.index >= len(replay_states):
                print("Finished replaying all states.")
                iface.close()
                return
            iface.rewind_to_state(replay_states[self.index])
            print(f"Step {self.index + 1}/{len(replay_states)}")

def main():
    client = ReplayClient()
    run_client(client)

if __name__ == "__main__":
    main()
