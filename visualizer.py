import pygame
import numpy as np
from multiprocessing import shared_memory
import math

# === Settings ===
WIDTH, HEIGHT = 600, 600
CENTER = (WIDTH // 2, HEIGHT // 2 + 200)
SCALE = 600  # pixels per meter

# === Connect to shared memory ===
shm = shared_memory.SharedMemory(name='tmdata')
# Infer num_beams by dividing shm size by 8 bytes per float64
num_beams = 13
data = np.ndarray((num_beams,), dtype=np.float64, buffer=shm.buf)

beam_angles_deg = [-60.0, -40.0, -20.0, -12.0, -7.0, -4.0, 0.0, 4.0,  7.0,  12.0, 20.0, 40.0, 60.0]

print(num_beams)
# === Pygame setup ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LiDAR Visualizer")
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((20, 20, 20))

    # Draw origin (car position)
    pygame.draw.circle(screen, (255, 255, 255), CENTER, 5)

    # Read LiDAR distances
    lidar = data[:num_beams]

    for i, dist in enumerate(lidar):
        angle_deg = beam_angles_deg[i]
        angle_rad = math.radians(angle_deg)

        # Flip X to correct horizontal mirroring consistent with original
        end_x = CENTER[0] - dist * SCALE * math.sin(angle_rad)
        end_y = CENTER[1] - dist * SCALE * math.cos(angle_rad)

        color = (255, 100, 100)
        pygame.draw.line(screen, color, CENTER, (end_x, end_y), 2)
        pygame.draw.circle(screen, color, (int(end_x), int(end_y)), 3)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
shm.close()
