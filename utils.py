import matplotlib.pyplot as plt
import pickle

action_space = {
    0: {'left': True, 'right': False, 'accelerate': False},
    1: {'left': True, 'right': False, 'accelerate': True},
    2: {'left': False, 'right': True, 'accelerate': False},
    3: {'left': False, 'right': True, 'accelerate': True},
    4: {'left': False, 'right': False, 'accelerate': True},
    5: {'left': False, 'right': False, 'accelerate': False}
}

def saveFile(name, object):
    with open(name, "wb") as f:
        pickle.dump(object, f)

def plot_data(values):
    plt.plot(values)
    plt.show()

def play_action(iface, final_action):
    for i in range(len(action_space)):
        if final_action[i] == 1:
            tmi_kwargs = action_space[i]
            
    iface.set_input_state(**tmi_kwargs)


class VehicleDataInputs:
    def __init__(self, speed, acceleration, turning_rate, distance_to_centerline, angle_to_centerline, next_curve_distance, next_curve_direction):
        self.speed = speed
        self.acceleration = acceleration
        self.turning_rate = turning_rate
        self.distance_to_centerline = distance_to_centerline #lateral disstance to centerline
        self.angle_to_centerline = angle_to_centerline
        self.next_curve_distance = next_curve_distance #distance to center of turning block
        self.next_curve_direction = next_curve_direction #1 = right, 0 = left



def generateCenterline(roadBlocks):
    inc = 0.5
    cl = []
    for i in range(len(roadBlocks)-1):
        c = roadBlocks[i]
        n = roadBlocks[i+1]
        if c[0] == n[0]:
            if c[1] > n[1]:
                y = c[1]
                while y > n[1]:
                    cl.append(( c[0], round(y,1) ))
                    y -= inc
            else:
                y = c[1]
                while y < n[1]:
                    cl.append(( c[0], round(y,1) ))
                    y += inc
        else:
            if c[0] > n[0]:
                x = c[0]
                while x > n[0]:
                    cl.append(( round(x,1), c[1] ))
                    x -= inc
            else:
                x = c[0]
                while x < n[0]:
                    cl.append(( round(x,1), c[1] ))
                    x += inc
    return cl



#Global variables
roadBlocks = [(528, 528), (528, 512), (528, 496), (528, 480), (512, 480), (496, 480), (480, 464), (480, 448), (480, 416), (480, 400), (464, 400), (432, 400), (432, 432), (416, 448), (400, 464), (400, 496), (416, 496), (432, 528), (432, 544), (432, 560), (448, 560), (464, 544), (480, 528), (496, 528), (512, 560), (528, 560), (544, 560), (560, 560), (576, 576), (576, 592), (592, 608), (608, 608), (624, 592), (640, 576), (656, 560), (656, 544), (640, 528), (608, 528), (592, 528)]
cl = generateCenterline(roadBlocks)
cornerBlockIndices = [0,3,6,11,14,17,19,22,24,28,30,32,34,36]

NUM_BLOCKS = len(roadBlocks)