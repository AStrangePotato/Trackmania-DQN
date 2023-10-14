import threading
import pygame
import time

pygame.init()
screen = pygame.display.set_mode((500, 500))
global state

def test():
    global carState
    import interface
    while True:
        time.sleep(0.01)
        carState = interface.getState()


t1 = threading.Thread(target=test)
t1.start()

print("started thread")
running = True
while running:   
    for event in pygame.event.get(): 
        
        # Check for QUIT event       
        if event.type == pygame.QUIT: 
            running = False

    screen.fill((0,0,0))
    state = interface.getState()
    pygame.draw.rect(screen, (255,255,225), pygame.Rect(state.position[0]%500, state.position[2]%500, 20,20))
    pygame.display.flip()
