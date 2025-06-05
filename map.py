import pygame
import os
import threading
import time
import math
import sys
import pickle
from utils import *

# Load centerline
c = generateCenterline(roadBlocks)

# Load checkpoints from states/rewards.sim
checkpoint_file = "states/rewards.sim"
checkpoints = []
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "rb") as f:
        checkpoints = pickle.load(f)
    print(f"Loaded {len(checkpoints)} checkpoints from {checkpoint_file}")
else:
    print(f"No checkpoint file found at {checkpoint_file}")

# Pygame setup
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (10, 50)
pygame.font.init()
my_font = pygame.font.SysFont('Comic Sans MS', 10)

scale = 0.9
rect_size = 10
offset = 220
screen = pygame.display.set_mode((1000, 1000))
running = True

def draw_rectangle(x, y, width, height, color, screen, line_width, rot_radians=0):
    points = []
    radius = math.sqrt((height / 2)**2 + (width / 2)**2)
    angle = math.atan2(height / 2, width / 2)
    angles = [angle, -angle + math.pi, angle + math.pi, -angle]
    for angle in angles:
        y_offset = -1 * radius * math.sin(angle + rot_radians)
        x_offset = radius * math.cos(angle + rot_radians)
        points.append((x + x_offset, y + y_offset))
    pygame.draw.polygon(screen, color, points, line_width)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            sys.exit()

    time.sleep(0.05)
    screen.fill((0, 0, 0))

    # Draw road blocks
    for i, r in enumerate(roadBlocks):
        scaled_x = int((r[0] - offset) * scale) + 200
        scaled_y = int((r[1] - offset) * scale) + 100
        text_surface = my_font.render(str(i), False, (255, 255, 255))
        color = (255, 0, 0) if i in c else (0, 255, 0)
        draw_rectangle(scaled_x, scaled_y, rect_size, rect_size, color, screen, 1)
        screen.blit(text_surface, (scaled_x, scaled_y))

    # Draw checkpoints as red dots
    for pos in checkpoints:
        x, y = pos
        scaled_x = int((x - offset) * scale) + 200
        scaled_y = int((y - offset) * scale) + 100
        pygame.draw.circle(screen, (255, 0, 0), (scaled_x, scaled_y), 3)

    pygame.display.update()
