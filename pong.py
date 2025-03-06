# linux
# source $HOME/projvenv/bin/activate linux
import numpy as np
import pygame
import time
import random
from math import sin , cos, pi




def reset_ball(resolution,ball_speed,p_serves = None):
   
    ball_pos = (0,0)
    ball_pos = (resolution[0] // 2, resolution[1] // 2)
    angle = np.random.uniform( pi/8,pi/4)
    if p_serves is None :
       player = random.choice([-1,1])
    y_sign =   random.choice([-1,1])   

    ball_dx = ball_speed*cos(angle)*player
    ball_dy = ball_speed*sin(angle)*y_sign
   
    return ball_pos,ball_dx,ball_dy

def compute_reward(ball_x,paddle_y,ball_y,resolution,paddle_length):
        
    if ball_x > 20 :
        if  paddle_y  <= ball_y <= min(paddle_y+10,resolution[1]):
            return 2
        elif  paddle_y+10  < ball_y < min(paddle_y + paddle_length -10, resolution[1]):
            return 1
        elif paddle_y + paddle_length -10  <= ball_y <= min(paddle_y + paddle_length , resolution[1]):
            return 2
        else:
            return  round(max(-min(abs(ball_y-paddle_y),abs(ball_y-(paddle_y+paddle_length))) * .01 ,-2),2)
    elif ball_x <= 20:
            if  paddle_y  <= ball_y <= min(paddle_y+10,resolution[1]):
                return 400
            elif  paddle_y+10  < ball_y < min(paddle_y + paddle_length -10, resolution[1]):
                return 200
            elif paddle_y + paddle_length -10  <= ball_y <= min(paddle_y + paddle_length , resolution[1]):
                return 400
            else: return -100
            


pygame.init()

#Training mode

#Constants
resolution = (640,480) # (width, height)
fps = 60
clock = pygame.time.Clock()
vs_human = True
paddle_size = (10,50)
ball_radius = 10
paddle_speed = 5
ball_speed = 3

#Palette RGB
c_background = (0,0,0)
c_players = (255,255,255)
c_ball = (255,255,255)

# Initialize paddles and ball coordinates
ball = pygame.Rect(resolution[0] // 2, resolution[1] // 2, ball_radius*2 , ball_radius*2 )
paddle1 = pygame.Rect(10, resolution[1] // 2 - paddle_size[1] // 2, paddle_size[0], paddle_size[1])
paddle2 = pygame.Rect(resolution[0] - 20, resolution[1] // 2 - paddle_size[1] // 2, paddle_size[0], paddle_size[1])
ball_pos,ball_dx, ball_dy = reset_ball(resolution,ball_speed,None)

#Display
screen = pygame.display.set_mode(resolution)
pygame.display.set_caption("Deep-Pong")

#render_game
training_mode = False
run = True

while run:

    dt = clock.tick(0) / 1000 if training_mode else clock.tick(15) / 1000  # No FPS limit when training
    dt = 1
    
  
    if not training_mode:
        screen.fill(c_background)


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
           run = False
    
    # Draw paddles and ball with new positions 

        # Paddle movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and paddle1.top > 0:
        paddle1.y -= paddle_speed * dt
        # print(paddle1.y,dt)
    if keys[pygame.K_s] and paddle1.bottom < resolution[1]:
        paddle1.y += paddle_speed * dt
        # print(paddle1.y,dt)
    if keys[pygame.K_UP] and paddle2.top > 0:
        paddle2.y -= paddle_speed * dt
    if keys[pygame.K_DOWN] and paddle2.bottom < resolution[1]:
        paddle2.y += paddle_speed * dt


    #Updating_ball
    ball.x += ball_dx * dt
    ball.y += ball_dy * dt
    ball_pos = (ball.x, ball.y) 
    
    # Checinkg for collisions with the walls
    #  Collition with top and bottom
    if ball.top <= 0 or ball.bottom >= resolution[1] :
        ball_dy *=-1
    # # Collition with Paddles
    # if ball.colliderect(paddle1) or ball.colliderect(paddle2):
    #     ball_dx *= -1

    if ball.colliderect(paddle1):  # Collision with left paddle
        offset = (ball.centery - paddle1.centery) / (paddle_size[1] / 2)  # Normalize impact point
        ball_dx = abs(ball_dx)  # Ensure ball moves right
        ball_dy = ball_speed * offset  # Change angle based on hit position

    if ball.colliderect(paddle2):  # Collision with right paddle
        offset = (ball.centery - paddle2.centery) / (paddle_size[1] / 2)
        ball_dx = -abs(ball_dx)  # Ensure ball moves left
        ball_dy = ball_speed * offset
        
    if ball.left <= 0 or ball.right >= resolution[0]:
       ball_pos,ball_dx, ball_dy = reset_ball(resolution,ball_speed)
       ball.x,ball.y = ball_pos

    reward=compute_reward(ball_pos[0],paddle1.y,ball_pos[1],resolution,paddle_size[1])
    print(f'top: {paddle1.y} , ball_y:{ball_pos[1]},bottom: {paddle1.y + 50}, reward: {reward} paddle_x+10:{paddle1.right} ball_x:{ball_pos[0]} ')





    if not training_mode :
        pygame.draw.rect(screen,c_players, paddle1)
        pygame.draw.rect(screen,c_players, paddle2,)
        pygame.draw.rect(screen,c_players, ball,1)
        pygame.draw.circle(screen,c_players,ball.center,ball_radius )
        pygame.display.flip()

    #time.sleep(5)
    #run =False






