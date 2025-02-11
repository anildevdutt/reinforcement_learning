import pygame
from rl import RL
import numpy as np

pygame.init()
screen = pygame.display.set_mode((1800, 720))
clock = pygame.time.Clock()
running = True
rl = RL()

xa_offset = 300 
ya_offset = 2
x_offset = 600
y_offset = 2
xb_offset = 2
yb_offset = 2
fx_offset = 2
fy_offset = 2
fy = 13
width = 55
height = 55
lx_offset = int(width/2)
ly_offset = int(height/8)
a_off = 2
frame = 0

arrow = pygame.Surface((50, 50))
arrow.fill("white")
pygame.draw.line(arrow, "red", (25,15), (25,45), 6)
pygame.draw.polygon(arrow, "red", ((35,15), (25,5), (15, 15)))


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("white")

    if frame%5 == 0:
        rl.run_episode()

    # actio value arrow table
    for i in range(len(rl.environment)):
        for j in range(len(rl.environment[0])):
            x = (j*width) + xb_offset
            y = (i*height) + yb_offset
            if rl.environment[i][j] == 1:
                pygame.draw.rect(screen, "black", (x, y, width, height))
            else:
                pygame.draw.rect(screen, "black", (x, y, width, height),  1)
    for t in rl.trajectory:
        x1 = (t["state"][1]*width) + xb_offset + int(width/2)
        y1 = (t["state"][0]*height) + yb_offset + int(height/2)
        x2 = (t["next_state"][1]*width) + xb_offset + int(width/2)
        y2 = (t["next_state"][0]*height) + yb_offset + int(height/2)
        pygame.draw.line(screen, "red", (x1, y1), (x2, y2), 2)



    # actio value arrow table
    for i in range(len(rl.environment)):
        for j in range(len(rl.environment[0])):
            x = (j*width) + xa_offset
            y = (i*height) + ya_offset
            pygame.draw.rect(screen, "black", (x, y, width, height),  1)
            action = np.argmax(rl.state_action_values[i][j]).item()
            if action == 0:
                screen.blit(arrow, (x+a_off, y+a_off))
            elif action == 1:
                screen.blit(pygame.transform.rotate(arrow, 270), (x+a_off, y+a_off))
            elif action == 2:
                screen.blit(pygame.transform.rotate(arrow, 180), (x+a_off, y+a_off))
            else:
                screen.blit(pygame.transform.rotate(arrow, 180), (x+a_off, y+a_off))


    # actio value table
    font = pygame.font.SysFont('monospace', 12)
    font.set_bold(True)
    for i in range(len(rl.environment)):
        for j in range(len(rl.environment[0])):
            x = (j*width) + x_offset
            y = (i*height) + y_offset
            pygame.draw.rect(screen, "black", (x, y, width, height),  1)
            text = font.render(f"U:{rl.state_action_values[i][j][0]:.2f}", True, "black")
            screen.blit(text, (fx_offset+x, fy_offset+y))

            text = font.render(f"R:{rl.state_action_values[i][j][1]:.2f}", True, "black")
            screen.blit(text, (fx_offset+x, fy_offset+y+fy))

            text = font.render(f"D:{rl.state_action_values[i][j][2]:.2f}", True, "black")
            screen.blit(text, (fx_offset+x, fy_offset+y+fy*2))

            text = font.render(f"L:{rl.state_action_values[i][j][3]:.2f}", True, "black")
            screen.blit(text, (fx_offset+x, fy_offset+y+fy*3))


    pygame.display.flip()
    clock.tick(60)
    frame += 1

pygame.quit()
