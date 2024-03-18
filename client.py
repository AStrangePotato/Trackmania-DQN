import socket
import struct
import time
import signal
from tminterface.structs import SimStateData, CheckpointData

HOST = "127.0.0.1"
PORT = 3141

SC_RUN_STEP_SYNC = 0
C_SET_SPEED = 1
C_REWIND_TO_STATE = 2
C_SET_INPUT_STATE = 3
C_SHUTDOWN = 4

sock = None

def signal_handler(sig, frame):
    global sock

    print('Shutting down...')
    sock.sendall(struct.pack('i', C_SHUTDOWN))
    sock.close()


def rewind_to_state(sock, state):
    sock.sendall(struct.pack('i', C_REWIND_TO_STATE))
    sock.sendall(struct.pack('i', len(state.data)))
    sock.sendall(state.data)

def set_input_state(sock, up=-1, down=-1, steer=0x7FFFFFFF):
    sock.sendall(struct.pack('i', C_SET_INPUT_STATE))
    sock.sendall(struct.pack('b', up))
    sock.sendall(struct.pack('b', down))
    sock.sendall(struct.pack('i', steer))

def respond(sock, type):
    sock.sendall(struct.pack('i', type))

def main():
    global sock

    first_state = 0
    now = time.time()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    signal.signal(signal.SIGINT, signal_handler)

    sock.connect((HOST, PORT))
    print(f'Connected to Port:{PORT}')
    while True:
        message_type = struct.unpack('i', sock.recv(4))[0]
        if message_type == SC_RUN_STEP_SYNC:
            state_length = struct.unpack('i', sock.recv(4))[0]
            state = SimStateData(sock.recv(state_length))


            respond(sock, SC_RUN_STEP_SYNC)

            if time.time() - now > 0.1:
                print(f"{state.position}")
                now = time.time()


if __name__ == "__main__":
    main()
