import math

action_space = {
    0: {'left': True, 'right': False, 'accelerate': False},
    1: {'left': True, 'right': False, 'accelerate': True},
    2: {'left': False, 'right': True, 'accelerate': False},
    3: {'left': False, 'right': True, 'accelerate': True},
    4: {'left': False, 'right': False, 'accelerate': False},
    5: {'left': False, 'right': False, 'accelerate': True}
}


roadBlocks = [(624, 528), (608, 528), (592, 528), (576, 528), (576, 512), (576, 496), (576, 480), (576, 464), (576, 448), (560, 448), (544, 448), (528, 448), (512, 448), (512, 432), (512, 416), (528, 416), (544, 416), (544, 400), (544, 384), (560, 384), (576, 384), (592, 384), (608, 384), (608, 400), (608, 416), (608, 432), (608, 448), (608, 464), (608, 480), (624, 480), (640, 480), (640, 464), (640, 448), (656, 448), (672, 448), (672, 432), (672, 416), (672, 400), (656, 400), (640, 400), (640, 384), (640, 368), (656, 368), (672, 368), (688, 368), (688, 352), (688, 336), (704, 336), (720, 336), (720, 352), (720, 368), (720, 384), (720, 400), (720, 416), (720, 432), (736, 432), (752, 432), (752, 448), (752, 464), (752, 480), (736, 480), (720, 480), (704, 480), (688, 480), (688, 496), (688, 512), (704, 512), (720, 512), (720, 528), (720, 544), (720, 560), (704, 560), (688, 560), (688, 576), (688, 592), (688, 608), (688, 624), (688, 640), (688, 656), (688, 672), (672, 672), (656, 672), (656, 656), (656, 640), (640, 640), (624, 640), (608, 640), (592, 640), (576, 640), (560, 640), (544, 640), (528, 640), (528, 624), (528, 608), (528, 592), (512, 592), (496, 592), (480, 592), (464, 592), (448, 592), (432, 592), (416, 592), (400, 592), (384, 592), (368, 592), (368, 576), (368, 560), (368, 544), (368, 528), (368, 512), (368, 496), (368, 480), (368, 464), (368, 448), (368, 432), (368, 416), (368, 400), (368, 384), (368, 368), (384, 368), (400, 368), (416, 368), (432, 368), (448, 368), (464, 368), (480, 368), (496, 368), (496, 352), (496, 336), (512, 336), (528, 336), (544, 336), (560, 336), (576, 336), (592, 336), (608, 336), (624, 336), (640, 336), (656, 336), (656, 320), (656, 304), (656, 288), (656, 272), (656, 256), (656, 240)]
cornerBlockIndices = [0,3,8,12,14,16,18,22,28,30,32,34,37,39,41,44,46,48,54,56,59,63,65,67,70,72,79,81,83,91,94,104,118,126,128,138]



def play_action(iface, final_action):
    for i in range(len(action_space)):
        if final_action[i] == 1:
            tmi_kwargs = action_space[i]
            
    iface.set_input_state(**tmi_kwargs)




class VehicleDataInputs:
    def __init__(self, speed, acceleration, turning_rate, lateral_velocity, distance_to_centerline, angle_to_centerline, next_curve_distance, next_curve_direction):
        self.speed = speed
        self.acceleration = acceleration
        self.turning_rate = turning_rate
        self.lateral_velocity = lateral_velocity
        self.distance_to_centerline = distance_to_centerline #lateral disstance to centerline
        self.angle_to_centerline = angle_to_centerline
        self.next_curve_distance = next_curve_distance #distance to center of turning block
        self.next_curve_direction = next_curve_direction #1 = right, 0 = left


def draw_rectangle(x, y, width, height, color, screen, line_width ,rot_radians=0):
    points = []

    radius = math.sqrt((height / 2)**2 + (width / 2)**2)

    angle = math.atan2(height / 2, width / 2)

    angles = [angle, -angle + math.pi, angle + math.pi, -angle]

    for angle in angles:
        y_offset = -1 * radius * math.sin(angle + rot_radians)
        x_offset = radius * math.cos(angle + rot_radians)
        points.append((x + x_offset, y + y_offset))

    pygame.draw.polygon(screen, color, points, line_width)
