import pygame
from multiprocessing import shared_memory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Connect to the shared memory created by main.py
existing_shm = shared_memory.SharedMemory(name='tmdata')  # Match the name from main.py
shared_data = np.ndarray((10,), dtype=np.float64, buffer=existing_shm.buf)

# Initialize Pygame
pygame.init()
offset = 1000
screen_width, screen_height = 1200, 900
zoom_factor = 3.0  # Zoom factor to enlarge the track and heatmap
scaled_width = int(screen_width)
scaled_height = int(screen_height)
screen = pygame.display.set_mode((scaled_width, scaled_height))  # Adjusted screen size for zoom
pygame.display.set_caption("Tracker")
clock = pygame.time.Clock()

# Create a list to store positions for heatmap effect
positions = []
max_points = 3000  # Store the most recent 3000 positions for heatmap effect

# Function to draw the heatmap effect
def draw_heatmap():
    if len(positions) > 0:
        # Create a surface with the heatmap effect (using matplotlib colormap)
        heatmap_surface = pygame.Surface((scaled_width, scaled_height), pygame.SRCALPHA)
        heatmap_surface.fill((0, 0, 0, 0))  # Transparent background

        # Create a scatter plot of positions for heatmap
        if len(positions) > max_points:
            positions.pop(0)  # Limit to max_points for performance reasons

        for pos in positions:
            # Map position to color using a colormap
            color_value = np.linalg.norm(pos)  # You can customize this
            color = plt.cm.viridis(Normalize()(color_value))  # Using the viridis colormap

            # Convert color from (r, g, b, a) to (r, g, b)
            r, g, b, a = color
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            alpha = int(a * 255)

            # Adjust the dot size to make them stack more
            radius = 5  # Adjust radius for more stacking

            # Scale the position based on zoom factor
            scaled_pos = [int(pos[0] * zoom_factor) - offset, int(pos[1] * zoom_factor)]

            # Draw a translucent dot (small and semi-transparent)
            pygame.draw.circle(heatmap_surface, (r, g, b, alpha), scaled_pos, radius)

        # Blit heatmap onto the main screen
        screen.blit(heatmap_surface, (0, 0))

# Function to draw modern, sleek sliders for agent state visualization
def draw_agent_state(agent_state):
    font = pygame.font.SysFont("Arial", 28)  # Larger font for better visibility
    labels = [
        ("Speed", agent_state[0]),
        ("Acceleration", agent_state[1]),
        ("Turning Rate", agent_state[2]),
        ("Distance to Centerline", agent_state[3]),
        ("Angle to Centerline", agent_state[4]),
        ("Next Curve Distance", agent_state[5]),
        ("Next Curve Direction", agent_state[6])
    ]
    bar_height = 30
    max_bar_width = 350  # Maximum width of the slider bars, wider for better visualization
    slider_height = 15  # Height of the slider bars
    label_margin = 20  # Margin between label and slider

    # Draw a semi-transparent background to clear the previous text space
    text_bg = pygame.Surface((max_bar_width + 20, len(labels) * (bar_height + 50)))  # Space for the bars and labels
    text_bg.fill((0, 0, 0, 128))  # Transparent black background
    screen.blit(text_bg, (10, 10))  # Blit the background to the screen at the starting position

    y_offset = 20  # Start position for the first label
    for label, value in labels:
        # Draw label text in white
        text = font.render(f"{label}: {value:.2f}", True, (255, 255, 255))  # White text
        screen.blit(text, (10, y_offset))  # Draw the label

        # Draw the sliders for each value
        normalized_value = max(0, min(1, value / 100.0))  # Normalize value to a 0-1 range (adjust if necessary)

        # Special handling for turning rate (negative to the left, positive to the right)
        if label == "Turning Rate":
            normalized_value = (value + 50) / 100  # Normalize so negative is on the left, positive on the right

        bar_width = int(normalized_value * max_bar_width)  # Scale the bar width according to the value
        pygame.draw.rect(screen, (50, 50, 50), (10, y_offset + label_margin + bar_height, max_bar_width, slider_height))  # Background bar (dark gray)
        pygame.draw.rect(screen, (0, 255, 0), (10, y_offset + label_margin + bar_height, bar_width, slider_height))  # Filled bar (green)

        # Add rounded corners for the slider bars
        pygame.draw.rect(screen, (255, 255, 255), (10, y_offset + label_margin + bar_height, max_bar_width, slider_height), 5)  # White outline with rounded corners

        y_offset += bar_height + 50  # Move down for the next label and slider

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read car's position and agent state from shared memory
    car_position = shared_data[:3]  # First 3 values for car position [x, y, z]
    agent_state = shared_data[3:]   # Next 7 values for agent state [speed, acceleration, turning_rate, ...]

    # Store the current position for heatmap effect
    positions.append([car_position[0], car_position[1]])

    # Draw the heatmap
    draw_heatmap()

    # Optionally, draw the current car position as a smaller, translucent dot
    # Scale the car's position based on the zoom factor
    scaled_car_x = int(car_position[0] * zoom_factor) - offset
    scaled_car_y = int(car_position[1] * zoom_factor)
    pygame.draw.circle(screen, (255, 0, 0, 150), (scaled_car_x, scaled_car_y), 5)  # Smaller, translucent dot

    # Draw the agent state sliders
    draw_agent_state(agent_state)

    # Update display
    pygame.display.flip()

    clock.tick(20)  # Limit to 20 FPS

# Clean up
pygame.quit()
existing_shm.close()
