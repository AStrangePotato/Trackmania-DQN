import math
from utils import *

def getAgentInputs(state, currentRoadBlockIndex, prevSpeed):
    a = (state.velocity[0]**2 + state.velocity[2]**2)**0.5
    b = a - prevSpeed
    c = state.scene_mobil.turning_rate
    d = getLateralVelocity(state, c)
    e = getDistanceToCenterLine(state, currentRoadBlockIndex)
    f = getAngleToCenterline(state, currentRoadBlockIndex)
    g = getDistanceToNextTurn(state, currentRoadBlockIndex)
    h = getNextTurnDirection(state, currentRoadBlockIndex)
    
    return [a,b,c,d,e,f,g,h]


def getCurrentRoadBlock(car_position):
    for i in range(len(roadBlocks)):
        if roadBlocks[i][0] - 8 < car_position[0] < roadBlocks[i][0] + 8: # x
            if roadBlocks[i][1] - 8 < car_position[2] < roadBlocks[i][1] + 8: # z
                return i

    #not on a road block
    return None

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

    
    return (angle_to_centerline + 3*pi) % (2 * pi) #making all positive


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




def getDistanceFromStart(position, current):
    lastCorner = 0
    for corner_i in cornerBlockIndices:
        if corner_i <= current:
            lastCorner = corner_i


    if lastCorner != current: #Not on a corner block
        if roadBlocks[current][0] == roadBlocks[lastCorner][0]: #z changes
            if roadBlocks[current][1] > roadBlocks[lastCorner][1]: #+z
                dist_from_last_turn = position[2] - roadBlocks[lastCorner][1]
            else:
                dist_from_last_turn = roadBlocks[lastCorner][1] - position[2]

        if roadBlocks[current][1] == roadBlocks[lastCorner][1]: #x changes
            if roadBlocks[current][0] > roadBlocks[lastCorner][0]: #+x
                dist_from_last_turn = position[0] - roadBlocks[lastCorner][0]
            else:
                dist_from_last_turn = roadBlocks[lastCorner][0] - position[0]
        dist_from_last_turn -= 8
    
    else: # on a corner
        nextBlock = current+1
        if roadBlocks[nextBlock][0] == roadBlocks[lastCorner][0]: #z changes
            if roadBlocks[nextBlock][1] > roadBlocks[lastCorner][1]: #+z
                p = roadBlocks[nextBlock][0], roadBlocks[nextBlock][1] - 8
            else:
                p = roadBlocks[nextBlock][0], roadBlocks[nextBlock][1] + 8
                
        if roadBlocks[nextBlock][1] == roadBlocks[lastCorner][1]: #x changes
            if roadBlocks[nextBlock][0] > roadBlocks[lastCorner][0]: #+x
                p = roadBlocks[nextBlock][0] - 8, roadBlocks[nextBlock][1]
            else:
                p = roadBlocks[nextBlock][0] + 8, roadBlocks[nextBlock][1]

        dist_from_last_turn = 16 - (abs(position[0]-p[0]) + abs(position[2]-p[1])) - 16
        
    total_dist = 16*lastCorner + dist_from_last_turn

    return total_dist

