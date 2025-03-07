import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
from pong_env import PongEnv  # Import our custom Pong environment
from datetime import datetime
import  os

print(tf.config.list_physical_devices('GPU'))
checkpoint_path = "checkpoints/pong_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create the environment
env = PongEnv(render_mode=False)

# Build the DQL model
model = Sequential([
    Dense(32, activation="relu", input_shape=(5,)),
    Dense(16, activation="relu"),
    Dense(env.action_space.n, activation="linear")
])
model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

model = tf.keras.models.load_model("my_model.keras")
# Experience Replay Memory
memory = deque(maxlen=20000)
gamma = 0.95  # Discount factor
epsilon = .01 # Exploration rate
epsilon_min = 0.01
epsilon_decay = 1
mini_batch_size = 64
rewards_per_episode = []
epsilon_history = []

# Training loop
for episode in range(1000):
 
    start_time = datetime.now()
    state = env.reset()
    time_reset = start_time-datetime.now()
    # print(f'Reset time: {(start_time-datetime.now()).total_seconds()}')
    state = tf.reshape(state, [1, 5])
    done = False
    total_reward = 0

    total_desiciontime = 0
    desicion = 'rand'
    
    start_time = datetime.now()
    while not done:
        start_time_des = datetime.now()
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()  # Explore , go up down or stay
        else:
            desicion='mod'
            action = np.argmax(model.predict(state, verbose=0))  # Exploit 
        total_desiciontime += (datetime.now()- start_time_des ).total_seconds()

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 5])
        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward
        
    rewards_per_episode.append(total_reward)
    epsilon_history.append(epsilon)
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
    if len(memory) > mini_batch_size:
        batch = random.sample(memory, mini_batch_size)

        # Convert to TensorFlow tensors for better GPU utilization
        batch_states = tf.convert_to_tensor([exp[0] for exp in batch], dtype=tf.float32)
        batch_next_states = tf.convert_to_tensor([exp[3] for exp in batch], dtype=tf.float32)

        # Batch predictions (happens on GPU if available)
        # reshape 
        batch_states = tf.squeeze(batch_states,1)
        batch_next_states =  tf.squeeze(batch_next_states,1)


        # Batch predictions (happens on GPU if available)
        batch_q_values = model(batch_states, training=False)  # Use model directly
        batch_next_q_values = model(batch_next_states, training=False)

        # Compute targets
        targets = []
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            target = reward
            if not done:
                target += gamma * tf.reduce_max(batch_next_q_values[i])  # Tensor operation
            target_f = batch_q_values[i].numpy()  # Convert to NumPy to update the action
            target_f[action] = target
            targets.append(target_f)

        # Convert to tensor
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)

        # Train once on the entire batch
        model.fit(batch_states, targets, epochs=1, verbose=0)




    time_random_s = start_time-datetime.now()
    print(f"Episode {episode} - Reward: {total_reward}")
    print(f"Reset time {round(time_reset.total_seconds()*1000,4)}ms,Desicion:time {desicion} {round(total_desiciontime,4)}s ,Episode time: {time_episode.total_seconds()} s - Random_s:{time_random_s.total_seconds()} s ")

    
    model.save('my_model.keras')

env.close()
