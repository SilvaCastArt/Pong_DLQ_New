import gymnasium as gym
import numpy as np
import pygame
import random

# We need to create a game enviroment
class PongEnv(gym.Env):
    def __init__(self):
        super(PongEnv,self).__init__()
        
        self.resolution = (640, 480)
        self.ball_speed = 3
        self.paddle_speed = 3
        self.action_space = gym.spaces.Discrete(5) # 5 actions: Stay, Up small, Down small, Up fast, Down fast
        self.observation_space = gym.spaces.Box()

        self.reset()
    
    def reset(self):
        self.ball_x = self.resolution[0] // 2
        self.ball_y = self.resolution[1] // 2