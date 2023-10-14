from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from trackData import *
import pygame
import sys
import threading
import time
import numpy as np
import os
#import agent


global interface_state
global agent_state
interface_state = None


os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (10, 50)

    
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

        
class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface: TMInterface, _time: int):
        global interface_state
        prevSpeed = 0
        prevDistance = 0
        
        if _time >= 0 and _time % 1 == 0:
            interface_state = iface.get_simulation_state()
            
            currentRoadBlockIndex = None
            car_position = interface_state.position
            for i in range(len(roadBlocks)):
                if roadBlocks[i][0] - 8 < car_position[0] < roadBlocks[i][0] + 8: # x
                    if roadBlocks[i][1] - 8 < car_position[2] < roadBlocks[i][1] + 8: # z
                        currentRoadBlockIndex = i
                        break

            if currentRoadBlockIndex is not None:
                agent_state = getAgentInputs(interface_state, currentRoadBlockIndex, prevSpeed)
                prevSpeed = agent_state.speed


                #Calculate reward
                currentDistance = getDistanceFromStart(interface_state, currentRoadBlockIndex)
                reward = currentDistance - prevDistance
                prevDistance = currentDistance
                
                print(reward)

                
            else:
                print("Off track")
            #iface.set_input_state(accelerate=True)

def pygameThread():
    offset = 220
    screen = pygame.display.set_mode((700,500))
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()
        
        time.sleep(0.01)

        visuals_carx = interface_state.position[0]-offset
        visuals_cary = interface_state.position[2]-offset
        
        screen.fill((0,0,0))
        
        for r in roadBlocks:
            draw_rectangle(r[0]-offset, r[1]-offset, 16,16, (255,0,0), screen, 1)

        draw_rectangle(visuals_carx, visuals_cary, 4, 7, (255,255,255), screen, 0, interface_state.yaw_pitch_roll[0])


            
        pygame.display.update()
        

t1 = threading.Thread(target=pygameThread)
t1.start()

server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
print(f'Connecting to {server_name}...')
run_client(MainClient(), server_name)


##        pos = state.position
##        for i in range(60):
##            for j in range(60):
##                if i*16-8 < pos[0] < i*16+8:
##                    if j*16-8 < pos[2] < j*16+8:
##                        blockCenter = (i*16, j*16)
##                        if blockCenter not in roadBlocks:
##                            roadBlocks.append(blockCenter)
##                            print(blockCenter)
##        for r in roadBlocks:
##            pygame.draw.rect(screen, (255,0,0), pygame.Rect(r[0]-offset, r[1]-offset, 8, 8))


