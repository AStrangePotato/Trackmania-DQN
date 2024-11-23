import pygame
import os
import threading
import time
import math
import sys


roadBlocks = [(528, 528), (528, 512), (528, 496), (528, 480), (512, 480), (496, 480), (480, 480), (480, 464), (480, 448), (480, 432), (480, 416), (480, 400), (464, 400), (448, 400), (432, 400), (432, 416), (432, 432), (432, 448), (416, 448), (400, 448), (400, 464), (400, 480), (400, 496), (416, 496), (432, 496), (432, 512), (432, 528), (432, 544), (432, 560), (448, 560), (464, 560), (464, 544), (464, 528), (480, 528), (496, 528), (496, 544), (496, 560), (512, 560), (528, 560), (544, 560)]
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

    screen.fill((0,0,0))

    #for c in cl:
    #    pygame.draw.rect(screen, (255, 0, 255), (c[0]-offset, c[1]-offset, 1,1))

    i = 0
    for r in roadBlocks:
        text_surface = my_font.render(str(i), False, (255,255,255))
        draw_rectangle(r[0]-offset, r[1]-offset, 16,16, (255,0,0), screen, 1)
        i += 1
        screen.blit(text_surface, (r[0]-offset, r[1]-offset))

    pygame.display.update()
