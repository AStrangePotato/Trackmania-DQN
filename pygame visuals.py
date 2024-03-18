import pygame
from utils import *
import os
import threading
import time
import math
import sys

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (10, 50)

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


def pygameThread():
    pygame.font.init()
    my_font = pygame.font.SysFont('Comic Sans MS', 12)
    offset = 220
    screen = pygame.display.set_mode((700,500))
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()
        
        time.sleep(0.01)

        #visuals_carx = interface_state.position[0]-offset
        #visuals_cary = interface_state.position[2]-offset
        
        screen.fill((0,0,0))

        for c in cl:
            pygame.draw.rect(screen, (255, 0, 255), (c[0]-offset, c[1]-offset, 1,1))

        i = 0
        for r in roadBlocks:
            text_surface = my_font.render(str(i), False, (255,255,255))
            draw_rectangle(r[0]-offset, r[1]-offset, 16,16, (255,0,0), screen, 1)
            i += 1
            screen.blit(text_surface, (r[0]-offset, r[1]-offset))

        #car rect
        #draw_rectangle(visuals_carx, visuals_cary, 4, 7, (255,255,255), screen, 0, interface_state.yaw_pitch_roll[0])

        pygame.display.update()




t1 = threading.Thread(target=pygameThread)
t1.start()

        
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

