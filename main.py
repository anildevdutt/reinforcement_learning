import pygame
from rl import RL

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
rl = RL()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("white")

    # actio value table
    font = pygame.font.Font('freesansbold.ttf', 8)

    x_offset = 10
    y_offset = 10
    width = 80
    height = 80
    for i in range(len(rl.environment)):
        for j in range(len(rl.environment[0])):
            x = (j*width) + x_offset
            y = (i*height) + y_offset
            pygame.draw.rect(screen, "black", (x, y, width, height),  1)
            text = font.render("test is some text", True, "black", "green")
            # text_rect = text.get_rect()
            # text_rect.center = (x // 2, y // 2)
            screen.blit(text, (x, y))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()