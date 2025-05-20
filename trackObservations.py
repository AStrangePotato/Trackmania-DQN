import math
from utils import *



def getAgentInputs(state, currentRoadBlockIndex, prevSpeed):
    a = (state.velocity[0]**2 + state.velocity[2]**2)**0.5
    b = a - prevSpeed
    c = state.scene_mobil.turning_rate
    d = getDistanceToCenterLine(state, currentRoadBlockIndex)
    e = getAngleToCenterline(state, currentRoadBlockIndex)
    f = getDistanceToNextTurn(state, currentRoadBlockIndex)
    g = getNextTurnDirection(state, currentRoadBlockIndex)
        
    # Normalize each feature
    a = a / 50  # Normalize speed
    b = b / 50  # Normalize speed difference (change in speed)
    c = c  # Normalize turning rate
    d = d / 8  # Normalize distance to centerline
    e = e / 3.14  # Normalize angle to centerline (between -1 and 1)
    f = f / 100  # Normalize distance to next turn
    g = g  # Direction is already categorical (0 or 1)

    return [a,b,c,d,e,f,g]
    

def getCurrentRoadBlock(car_position):
    width = 8
    for i in range(len(roadBlocks)):
        if roadBlocks[i][0] - width < car_position[0] < roadBlocks[i][0] + width: # x
            if roadBlocks[i][1] - width < car_position[2] < roadBlocks[i][1] + width: # z
                return i

    #not on a road block
    return None

def getCenterlineEndblock(currentRoadBlockIndex):
    nextCorner = currentRoadBlockIndex + 1
    centerline_end_block = roadBlocks[-1]
    while nextCorner < NUM_BLOCKS:
        if nextCorner in cornerBlockIndices:
            centerline_end_block = roadBlocks[nextCorner]
            break
        else:
            nextCorner += 1

    return centerline_end_block

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

    centerline_end_block = getCenterlineEndblock(currentRoadBlockIndex)
    
    pi = 3.141592
    
    yaw = state.yaw_pitch_roll[0]

    #Case 1: road continues in the x direction -> z stays the same
    if currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex + 1][1]:
        if centerline_end_block[0] > currentBlockCenter[0]: #if the road continues +x direction
            angle_to_centerline = yaw - 1.571
        else: #road continues -x direction
            angle_to_centerline = yaw + 1.571

    #Case 2: road continues in the z direction -> x stays the same
    if currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex + 1][0]:
        if centerline_end_block[1] > currentBlockCenter[1]: #if the road continues +z direction
            angle_to_centerline = yaw
        else: #road continues -z direction
            angle_to_centerline = yaw + pi

    angle_to_centerline = (angle_to_centerline + 2*pi) % (2 * pi) #making all positive
    if angle_to_centerline < pi:
        return angle_to_centerline
    else:
        return angle_to_centerline - 2*pi

def getNextTurnDirection(state, currentRoadBlockIndex):
    currentBlockCenter = roadBlocks[currentRoadBlockIndex]
    centerline_end_block = getCenterlineEndblock(currentRoadBlockIndex)

    #Case 1: road continues in the x direction -> z stays the same
    if currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex + 1][1]:
        if centerline_end_block[0] > currentBlockCenter[0]: #if the road continues +x direction
            if roadBlocks[roadBlocks.index(centerline_end_block) + 1][1] > currentBlockCenter[1]:
                next_turn_direction = 1
            else:
                next_turn_direction = -1
        else: #road continues -x direction
            if roadBlocks[roadBlocks.index(centerline_end_block) + 1][1] > currentBlockCenter[1]:
                next_turn_direction = -1
            else:
                next_turn_direction = 1

    #Case 2: road continues in the z direction -> x stays the same
    if currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex + 1][0]:
        if centerline_end_block[1] > currentBlockCenter[1]: #if the road continues +z direction
            if roadBlocks[roadBlocks.index(centerline_end_block) + 1][0] > currentBlockCenter[0]:
                next_turn_direction = -1
            else:
                next_turn_direction = 1
        else: #road continues -x direction
            if roadBlocks[roadBlocks.index(centerline_end_block) + 1][0] > currentBlockCenter[0]:
                next_turn_direction = 1
            else:
                next_turn_direction = -1
                
    return next_turn_direction

def getDistanceToNextTurn(state, currentRoadBlockIndex):
    currentBlockCenter = roadBlocks[currentRoadBlockIndex]
    centerline_end_block = getCenterlineEndblock(currentRoadBlockIndex)

    #Case 1: road continues in the x direction -> z stays the same
    if currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex + 1][1]:
        if centerline_end_block[0] > currentBlockCenter[0]: #if the road continues +x direction
            dist_to_next_turn = centerline_end_block[0] - state.position[0]

        else: #road continues -x direction
            dist_to_next_turn = state.position[0] - centerline_end_block[0]
        
    #Case 2: road continues in the z direction -> x stays the same
    if currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex + 1][0]:
        if centerline_end_block[1] > currentBlockCenter[1]: #if the road continues +z direction
            dist_to_next_turn = centerline_end_block[1] - state.position[2]

        else: #road continues -z direction
            dist_to_next_turn = state.position[2] - centerline_end_block[1]

    return dist_to_next_turn

def getDistanceToCenterLine(state, currentRoadBlockIndex):
    if currentRoadBlockIndex not in cornerBlockIndices:
        currentBlockCenter = roadBlocks[currentRoadBlockIndex]

        #Case 1: road continues in the x direction -> z stays the same
        if currentBlockCenter[1] == roadBlocks[currentRoadBlockIndex + 1][1]:
            dist_to_centerline = abs(currentBlockCenter[1] - state.position[2])

        #Case 2: road continues in the z direction -> x stays the same
        elif currentBlockCenter[0] == roadBlocks[currentRoadBlockIndex + 1][0]:
            dist_to_centerline = abs(currentBlockCenter[0] - state.position[0])

    else: #on a corner
        dist_to_centerline = getClosestCenterlinePoint(state.position, currentRoadBlockIndex, True)

    return dist_to_centerline

def dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def getClosestCenterlinePoint(position, current, return_dist=False):
    position = (position[0], position[2])
    start = current * int(16 / 0.5) + 10
    closest = 999
    closestIndex = 0

    if current == 0:
        for i in range(0, 45): #random number that is definitely greater than the range
            d = dist(position, cl[i])
            if d < closest:
                closest = d
                closestIndex = i

    else:
        for i in range(start-16, start + 16):
            d = dist(position, cl[i])
            if d < closest:
                closest = d
                closestIndex = i

    if return_dist:
        return closest

    return closestIndex

