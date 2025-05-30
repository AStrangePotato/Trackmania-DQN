import math
from utils import *



def getAgentInputs(state, currentRoadBlockIndex):
    a = state.display_speed
    c = state.scene_mobil.turning_rate
    d = getDistanceToCenterLine(state, currentRoadBlockIndex)
    e = getAngleToCenterline(state, currentRoadBlockIndex)
    f = getDistanceToNextTurn(state, currentRoadBlockIndex)
    g = getNextTurnDirection(currentRoadBlockIndex)
    
    # Normalize features
    a /= 130      # speed normalized
    d /= 8       # distance to centerline normalized
    e /= 3.14    # angle to centerline normalized (-1 to 1)
    f /= 100     # distance to next turn normalized
    
    return [a, c, d, e, f, g]

import math

def simulate_lidar_raycast(state,
                           currentRoadBlockIndex,
                           num_beams=12,
                           max_range=60.0,
                           step=0.2):
    px, _, pz = state.position
    yaw = state.yaw_pitch_roll[0] % (2 * math.pi)

    beam_angles_deg = [-60.0, -40.0, -20.0, -10.0, -5.0, -2.0,
                        2.0,  5.0,  10.0, 20.0, 40.0, 60.0]
    directions = [
        (math.sin(yaw + math.radians(a)), math.cos(yaw + math.radians(a)))
        for a in beam_angles_deg
    ]

    candidates = roadBlocks[currentRoadBlockIndex : currentRoadBlockIndex + 3]
    half_block = 8.0  # half of 16

    def inside_any_block(x, z):
        return any(abs(x - cx) <= half_block and abs(z - cz) <= half_block for cx, cz in candidates)

    distances = []
    for dx, dz in directions:
        for i in range(1, int(max_range / step) + 1):
            dist = i * step
            sx = px + dx * dist
            sz = pz + dz * dist

            if not inside_any_block(sx, sz):
                distances.append(dist)
                break
        else:
            distances.append(max_range)

    return distances



def getCurrentRoadBlock(car_position, guess=0):
    width = 8
    n = NUM_BLOCKS
    
    # Check the guess index first
    if 0 <= guess < n:
        if roadBlocks[guess][0] - width < car_position[0] < roadBlocks[guess][0] + width:  # x
            if roadBlocks[guess][1] - width < car_position[2] < roadBlocks[guess][1] + width:  # z
                return guess
    
    # Alternate checking one forward and one backward from guess
    for offset in range(1, n):
        # Check backward index (guess - offset)
        backward = guess - offset
        if backward >= 0:
            if roadBlocks[backward][0] - width < car_position[0] < roadBlocks[backward][0] + width:  # x
                if roadBlocks[backward][1] - width < car_position[2] < roadBlocks[backward][1] + width:  # z
                    return backward
        
        # Check forward index (guess + offset)
        forward = guess + offset
        if forward < n:
            if roadBlocks[forward][0] - width < car_position[0] < roadBlocks[forward][0] + width:  # x
                if roadBlocks[forward][1] - width < car_position[2] < roadBlocks[forward][1] + width:  # z
                    return forward

    # No roadblock found
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

def getNextTurnDirection(currentRoadBlockIndex):
    currentBlockCenter = roadBlocks[currentRoadBlockIndex]
    centerline_end_block = getCenterlineEndblock(currentRoadBlockIndex)

    if centerline_end_block == roadBlocks[-1]:
        return 1
    
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

def getClosestCenterlinePoint(position, roadBlockIndex, return_dist=False):
    px, _, pz = position
    pos2d = (px, pz)

    closest_dist = float('inf')
    closest_idx  = 0

    for idx in range(NUM_CP):
        d = dist(pos2d, cp[idx])
        if d < closest_dist:
            closest_dist = d
            closest_idx  = idx

    return closest_dist if return_dist else closest_idx

