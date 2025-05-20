import pygame
from multiprocessing import shared_memory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Shared memory: expects 10 floats [x, y, z, speed, accel, turn_rate, dist_center, angle_center, next_curve_dist, next_curve_dir]
shm = shared_memory.SharedMemory(name='tmdata')
data = np.ndarray((10,), dtype=np.float64, buffer=shm.buf)

# Pygame setup
pygame.init()
W, H, ZOOM, OFFSET = 800, 600, 2.0, 500
screen = pygame.display.set_mode((W, H))
clock = pygame.time.Clock()

# Visualization buffers
positions, MAX_POS = [], 2000
font = pygame.font.SysFont("Arial", 20)

def norm_color(val):  # Normalize & map to colormap
    color = plt.cm.viridis(Normalize()(val))
    return tuple(int(c * 255) for c in color[:3]) + (int(color[3] * 180),)

def draw_heatmap():
    surf = pygame.Surface((W, H), pygame.SRCALPHA)
    for pos in positions[-MAX_POS:]:
        x, y = int(pos[0] * ZOOM) - OFFSET, int(pos[1] * ZOOM)
        pygame.draw.circle(surf, norm_color(np.linalg.norm(pos)), (x, y), 4)
    screen.blit(surf, (0, 0))

def draw_state(state):
    labels = ["Speed", "Accel", "Turn", "Dist", "Angle", "NextDist", "NextDir"]
    base_y = 10
    for i, (label, val) in enumerate(zip(labels, state)):
        y = base_y + i * 40
        text = font.render(f"{label}: {val:.2f}", True, (255, 255, 255))
        screen.blit(text, (10, y))
        norm_val = (val + 50) / 100 if label == "Turn" else max(0, min(val, 1))
        pygame.draw.rect(screen, (50, 50, 50), (150, y + 5, 200, 10))
        pygame.draw.rect(screen, (0, 255, 0), (150, y + 5, int(norm_val * 200), 10))

# Main loop
running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))
    pos = data[:3]; state = data[3:]
    positions.append([pos[0], pos[1]])

    draw_heatmap()
    x, y = int(pos[0] * ZOOM) - OFFSET, int(pos[2] * ZOOM)
    pygame.draw.circle(screen, (255, 0, 0), (x, y), 5)
    draw_state(state)

    pygame.display.flip()
    clock.tick(20)

pygame.quit()
shm.close()
