import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
from pong_env import PongEnv  # Import our custom Pong environment
from datetime import datetime

print(tf.config.list_physical_devices('GPU'))

# Create the environment
env = PongEnv(render_mode=True)

# Build the DQL model
model = Sequential([
    Dense(24, activation="relu", input_shape=(5,)),
    Dense(24, activation="relu"),
    Dense(env.action_space.n, activation="linear")
])
model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Experience Replay Memory
memory = deque(maxlen=2000)
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

# Training loop
for episode in range(1000):

    start_time = datetime.now()
    state = env.reset()
    time_reset = start_time-datetime.now()
    # print(f'Reset time: {(start_time-datetime.now()).total_seconds()}')
    state = np.reshape(state, [1, 5])
    done = False
    total_reward = 0
    
    start_time = datetime.now()
    while not done:
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(model.predict(state, verbose=0))  # Exploit

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 5])
        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward
    time_episode = start_time-datetime.now()


    # print(f"Episode {episode} - Reward: {total_reward}")

    # Reduce exploration
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    start_time = datetime.now()
    # Train with random samples
    # if len(memory) > 32:
    #     batch = random.sample(memory, 32)
    #     for state, action, reward, next_state, done in batch:
    #         target = reward
    #         if not done:
    #             target += gamma * np.amax(model.predict(next_state, verbose=0))
    #         target_f = model.predict(state, verbose=0)
    #         target_f[0][action] = target
    #         model.fit(state, target_f, epochs=1, verbose=0)
    
    
    # Train with random samples
    if len(memory) > 32:
        batch = random.sample(memory, 32)
        batch_states = tf.convert_to_tensor([exp[0] for exp in batch], dtype=tf.float32)
        batch_next_states = tf.convert_to_tensor([exp[3] for exp in batch], dtype=tf.float32)




    time_random_s = start_time-datetime.now()

    print(f"Episode {episode} - Reward: {total_reward}")
    print(f"Reset time {time_reset.total_seconds()*1000}ms, Episode time: {time_episode.total_seconds()} s - Random_s:{time_random_s.total_seconds()} s ")

    
    #model.save('my_model.keras')

env.close()
