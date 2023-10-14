import math
import pygame


roadBlocks = [(624, 528), (608, 528), (592, 528), (576, 528), (576, 512), (576, 496), (576, 480), (576, 464), (576, 448), (560, 448), (544, 448), (528, 448), (512, 448), (512, 432), (512, 416), (528, 416), (544, 416), (544, 400), (544, 384), (560, 384), (576, 384), (592, 384), (608, 384), (608, 400), (608, 416), (608, 432), (608, 448), (608, 464), (608, 480), (624, 480), (640, 480), (640, 464), (640, 448), (656, 448), (672, 448), (672, 432), (672, 416), (672, 400), (656, 400), (640, 400), (640, 384), (640, 368), (656, 368), (672, 368), (688, 368), (688, 352), (688, 336), (704, 336), (720, 336), (720, 352), (720, 368), (720, 384), (720, 400), (720, 416), (720, 432), (736, 432), (752, 432), (752, 448), (752, 464), (752, 480), (736, 480), (720, 480), (704, 480), (688, 480), (688, 496), (688, 512), (704, 512), (720, 512), (720, 528), (720, 544), (720, 560), (704, 560), (688, 560), (688, 576), (688, 592), (688, 608), (688, 624), (688, 640), (688, 656), (688, 672), (672, 672), (656, 672), (656, 656), (656, 640), (640, 640), (624, 640), (608, 640), (592, 640), (576, 640), (560, 640), (544, 640), (528, 640), (528, 624), (528, 608), (528, 592), (512, 592), (496, 592), (480, 592), (464, 592), (448, 592), (432, 592), (416, 592), (400, 592), (384, 592), (368, 592), (368, 576), (368, 560), (368, 544), (368, 528), (368, 512), (368, 496), (368, 480), (368, 464), (368, 448), (368, 432), (368, 416), (368, 400), (368, 384), (368, 368), (384, 368), (400, 368), (416, 368), (432, 368), (448, 368), (464, 368), (480, 368), (496, 368), (496, 352), (496, 336), (512, 336), (528, 336), (544, 336), (560, 336), (576, 336), (592, 336), (608, 336), (624, 336), (640, 336), (656, 336), (656, 320), (656, 304), (656, 288), (656, 272), (656, 256), (656, 240)]

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

def getAgentInputs(state, currentRoadBlockIndex, prevSpeed):
    a = (state.velocity[0]**2 + state.velocity[2]**2)**0.5
    b = a - prevSpeed
    c = state.scene_mobil.turning_rate
    d = getLateralVelocity(state, c)
    e = getDistanceToCenterLine(state, currentRoadBlockIndex)
    f = getAngleToCenterline(state, currentRoadBlockIndex)
    g = getDistanceToNextTurn(state, currentRoadBlockIndex)
    h = getNextTurnDirection(state, currentRoadBlockIndex)
    
    return VehicleDataInputs(a,b,c,d,e,f,g,h)



def getLateralVelocity(state, turning_rate):
    pi = 3.14159
    if turning_rate > 0: #right
        angle = state.yaw_pitch_roll[0] + pi/2
    elif turning_rate < 0:
        angle = state.yaw_pitch_roll[0] - pi/2
    else:
        return 0
    
    
    lateral_unit_vector = [math.cos(angle), math.sin(angle)]
    mag = (lateral_unit_vector[0]**2 + lateral_unit_vector[1]**2)**0.5
    lateral_unit_vector[0] /= mag
    lateral_unit_vector[1] /= mag

    velocity_vector = [state.velocity[0], state.velocity[2]]

    projection = lateral_unit_vector[0] * velocity_vector[0] + lateral_unit_vector[1] * velocity_vector[1]
    
    return abs(projection) #lateral velocity
    
def getAngleToCenterline(state, currentRoadBlockIndex):
    currentBlockCenter = roadBlocks[currentRoadBlockIndex]
    pi = 3.141592
    yaw = state.yaw_pitch_roll[0]
    
    #Case 1: road continues in the x direction -> z stays the same
    if currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex + 1][1]:
        i = 0
        while currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex + i][1]:
            i += 1

        centerline_end_block = roadBlocks[currentRoadBlockIndex + i-1]
        if centerline_end_block[0] > currentBlockCenter[0]: #if the road continues +x direction
            angle_to_centerline = yaw - 1.571
        else: #road continues -x direction
            angle_to_centerline = yaw + 1.571

    #Case 2: road continues in the z direction -> x stays the same
    if currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex + 1][0]:
        i = 0
        while currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex + i][0]:
            i += 1

        centerline_end_block = roadBlocks[currentRoadBlockIndex + i-1]
        if centerline_end_block[1] > currentBlockCenter[1]: #if the road continues +z direction
            angle_to_centerline = yaw
        else: #road continues -z direction
            angle_to_centerline = yaw + pi

    
    return (angle_to_centerline + 2*pi) % (2 * pi) #making all positive


def getNextTurnDirection(state, currentRoadBlockIndex):
    currentBlockCenter = roadBlocks[currentRoadBlockIndex]
    #Case 1: road continues in the x direction -> z stays the same
    if currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex + 1][1]:
        i = 0
        while currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex + i][1]:
            i += 1

        centerline_end_block = roadBlocks[currentRoadBlockIndex + i-1]
        if centerline_end_block[0] > currentBlockCenter[0]: #if the road continues +x direction
            if roadBlocks[currentRoadBlockIndex + i][1] > currentBlockCenter[1]:
                next_turn_direction = 1
            else:
                next_turn_direction = -1
        else: #road continues -x direction
            if roadBlocks[currentRoadBlockIndex + i][1] > currentBlockCenter[1]:
                next_turn_direction = -1
            else:
                next_turn_direction = 1

    #Case 2: road continues in the z direction -> x stays the same
    if currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex + 1][0]:
        i = 0
        while currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex + i][0]:
            i += 1

        centerline_end_block = roadBlocks[currentRoadBlockIndex + i-1]
        if centerline_end_block[1] > currentBlockCenter[1]: #if the road continues +z direction
            if roadBlocks[currentRoadBlockIndex + i][0] > currentBlockCenter[0]:
                next_turn_direction = -1
            else:
                next_turn_direction = 1
        else: #road continues -x direction
            if roadBlocks[currentRoadBlockIndex + i][0] > currentBlockCenter[0]:
                next_turn_direction = 1
            else:
                next_turn_direction = -1
                
    return next_turn_direction


def getDistanceToNextTurn(state, currentRoadBlockIndex):
    currentBlockCenter = roadBlocks[currentRoadBlockIndex]
    #Case 1: road continues in the x direction -> z stays the same
    if currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex + 1][1]:
        i = 0
        while currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex + i][1]:
            i += 1

        centerline_end_block = roadBlocks[currentRoadBlockIndex + i-1]
        if centerline_end_block[0] > currentBlockCenter[0]: #if the road continues +x direction
            dist_to_next_turn = centerline_end_block[0] - state.position[0]
            facing_pos = True
            
        else: #road continues -x direction
            dist_to_next_turn = state.position[0] - centerline_end_block[0]
            facing_pos = False
        
    #Case 2: road continues in the z direction -> x stays the same
    if currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex + 1][0]:
        i = 0
        while currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex + i][0]:
            i += 1

        centerline_end_block = roadBlocks[currentRoadBlockIndex + i-1]
        if centerline_end_block[1] > currentBlockCenter[1]: #if the road continues +z direction
            dist_to_next_turn = centerline_end_block[1] - state.position[2]
            facing_pos = True
            
        else: #road continues -z direction
            dist_to_next_turn = state.position[2] - centerline_end_block[1]
            facing_pos = False
        
    return dist_to_next_turn


def getDistanceToCenterLine(state, currentRoadBlockIndex):
    currentBlockCenter = roadBlocks[currentRoadBlockIndex]
    #Case 1: road continues in the x direction -> z stays the same
    if currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex + 1][1]:
        dist_to_centerline = abs(currentBlockCenter[1] - state.position[2])

    #Case 2: road continues in the z direction -> x stays the same
    if currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex + 1][0]:
        dist_to_centerline = abs(currentBlockCenter[0] - state.position[0])

       
    return dist_to_centerline



def getDistanceFromStart(state, currentRoadBlockIndex):
    if currentRoadBlockIndex == 0:
        return abs(state.position[0] - 628.8)
    
    currentBlockCenter = roadBlocks[currentRoadBlockIndex]
    #Case 1: road continues in the x direction -> z stays the same
    if currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex - 1][1]:
        i = 0
        while currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex - i - 1][1]:
            i += 1

        travelled = state.position[0] - roadBlocks[currentRoadBlockIndex - i][0]
        
    #Case 2: road continues in the z direction -> x stays the same
    if currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex - 1][0]:
        i = 0
        while currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex - i - 1][0]:
            i += 1

        travelled = state.position[2] - roadBlocks[currentRoadBlockIndex - i][1]

    
    return currentRoadBlockIndex*16 + travelled


#misc functions
def draw_rectangle(x, y, width, height, color, screen, line_width ,rot_radians=0):
    """Draw a rectangle, centered at x, y.

    Arguments:
      x (int/float):
        The x coordinate of the center of the shape.
      y (int/float):
        The y coordinate of the center of the shape.
      width (int/float):
        The width of the rectangle.
      height (int/float):
        The height of the rectangle.
      color (str):
        Name of the fill color, in HTML format.
    """
    points = []

    # The distance from the center of the rectangle to
    # one of the corners is the same for each corner.
    radius = math.sqrt((height / 2)**2 + (width / 2)**2)

    # Get the angle to one of the corners with respect
    # to the x-axis.
    angle = math.atan2(height / 2, width / 2)

    # Transform that angle to reach each corner of the rectangle.
    angles = [angle, -angle + math.pi, angle + math.pi, -angle]

    # Calculate the coordinates of each point.
    for angle in angles:
        y_offset = -1 * radius * math.sin(angle + rot_radians)
        x_offset = radius * math.cos(angle + rot_radians)
        points.append((x + x_offset, y + y_offset))

    pygame.draw.polygon(screen, color, points, line_width)
