import gymnasium as gym
import numpy as np
import pygame
from math import pi, cos , sin



class PongEnv(gym.Env):
    def __init__(self, render_mode=False):
        super().__init__()
        self.render_mode = render_mode
        self.resolution = (640,480)
        self.side = -1 # which side the agent will defend right or left -1,1
        self.paddle_length = 50
        self.paddle_width = 10
        
        # Action space: 5 discrete actions (Stay, Small Up,, Small Down)
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: Paddle Y position, Ball X & Y position, Ball Velocity
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, -5, -5]), 
            high=np.array([480, 640, 480, 5, 5]),
            dtype=np.float32
        )

        pygame.init()
        self.screen = pygame.display.set_mode( self.resolution)
        self.screen.fill((0, 0, 0))

        # Initialize Pygame
        # pygame.init()
        # self.screen = pygame.display.set_mode((640, 480))
        # clock = pygame.time.Clock()
        # self.fps = clock.tick(0) 

        self.reset()

    def reset(self):
        """Reset game state."""
        self.paddle_y =  np.random.uniform(51, self.resolution[1]-51)
        self.ball_y = np.random.random_integers((self.resolution[1] // 2)-50,(self.resolution[1] // 2)+50)
        self.ball_x = self.resolution[0] // 2

        angle = np.random.uniform( pi/8,pi/4)
        angle_sign = np.random.choice([-1,1]) 
        speed = np.random.random_integers(3, 6)

        self.ball_dx = speed*cos(angle)*self.side
        self.ball_dy = speed*sin(angle)*angle_sign



        return np.array([self.paddle_y, self.ball_x, self.ball_y, self.ball_dx, self.ball_dy], dtype=np.float32)

    def step(self, action):
        """Apply action and update the environment."""
        if action == 1:  # Small Up
            self.paddle_y -= 5  
        elif action == 2:  # Small Down
            self.paddle_y += 5  

        self.paddle_y = np.clip(self.paddle_y, 0,  self.resolution[1])

        # Ball movement
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Collision with walls
        if self.ball_y <= 0 or self.ball_y >=  self.resolution[1]:
            self.ball_dy *= -1

        # Collision with left paddle (AI's paddle)
        if self.ball_x <= 20 and abs(self.paddle_y - self.ball_y) < 50:
            self.ball_dx *= -1

        # Reward system
        reward = self.compute_reward()

        # Check if ball went out (end of round)
        # If it touches the bar the game stops
        done = self.ball_x <= 20 

        # Render if enabled
        if self.render_mode:
            self.render()

        return np.array([self.paddle_y, self.ball_x, self.ball_y, self.ball_dx, self.ball_dy], dtype=np.float32), reward, done, {}

    def render(self):
        """Draw game objects on screen."""
                # Initialize Pygame

        clock = pygame.time.Clock()
        clock.tick(0) 
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), (10, self.paddle_y, self.paddle_width, self.paddle_length))
        # pygame.draw.rect(self.screen, (255, 255, 255), (6, self.paddle_y, self.paddle_width, self.paddle_length))
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y)), 5)
        pygame.display.flip()

    def compute_reward(self):
         
        if self.ball_x > 60 :
            if  self.paddle_y  <= self.ball_y <= min(self.paddle_y+10,self.resolution[1]):
                return 2
            elif  self.paddle_y+10  < self.ball_y < min(self.paddle_y + self.paddle_length -10, self.resolution[1]):
                return 1
            elif self.paddle_y + self.paddle_length -10  <= self.ball_y <= min(self.paddle_y + self.paddle_length , self.resolution[1]):
                return 2
            else:
                return max(-min(abs(self.ball_y-self.paddle_y),abs(self.ball_y-(self.paddle_y+self.paddle_length)))*.01,-2)
        elif self.ball_x <= 60:
            if  self.paddle_y  <= self.ball_y <= min(self.paddle_y+10,self.resolution[1]):
                return 200
            elif  self.paddle_y+10  < self.ball_y < min(self.paddle_y + self.paddle_length -10, self.resolution[1]):
                return 100
            elif self.paddle_y + self.paddle_length -10  <= self.ball_y <= min(self.paddle_y + self.paddle_length , self.resolution[1]):
                return 200
            else: return -100
        
            

        
