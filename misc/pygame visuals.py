import pygame
import os
import threading
import time
import math
import sys
from utils import *

c = generateCenterline(roadBlocks)

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

pygame.font.init()
my_font = pygame.font.SysFont('Comic Sans MS', 10)  # Smaller font

scale = 0.75  # Scale factor for position
rect_size = 8  # Smaller rectangle size
offset = 220

screen = pygame.display.set_mode((1000, 1000))  # Optional: smaller screen
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            sys.exit()
    
    time.sleep(0.05)
    screen.fill((0,0,0))

    for i, r in enumerate(roadBlocks):
        scaled_x = int((r[0] - offset) * scale) + 200
        scaled_y = int((r[1] - offset) * scale) + 100

        text_surface = my_font.render(str(i), False, (255, 255, 255))

        if i in c:
            draw_rectangle(scaled_x, scaled_y, rect_size, rect_size, (255, 0, 0), screen, 1)
        else:
            draw_rectangle(scaled_x, scaled_y, rect_size, rect_size, (0, 255, 0), screen, 1)

        screen.blit(text_surface, (scaled_x, scaled_y))
    pygame.display.update()
