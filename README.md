Deep Q-Learning (DQL) Algorithm Overview

Deep Q-Learning (DQL) is an extension of Q-Learning, which is a reinforcement learning algorithm. 
Instead of using a table to store Q-values, DQL uses a neural network to approximate the Q-values for different actions.

In this specific case I will train a neural network to play the classic game pong.
For training is only necessary the files pong_env.py as this file contains the environment,
and train_dql.py which contains the training algorithm. 

In train dql is possible to change env = PongEnv(render_mode=False) to True , to see how the algorithm performs


