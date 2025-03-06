import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import gymnasium as gym
from pong_env import PongEnv 

# Load the trained model

model = tf.keras.models.load_model('my_model.keras')
# Create the Pong environment
env = gym.make("Pong-v4", render_mode="human")  # Renders game visually
state, _ = env.reset()

done = False
while not done:
    # Preprocess state if needed (e.g., resizing, normalizing)
    state = np.expand_dims(state, axis=0)  # Add batch dimension

    # Agent selects action based on trained model
    action = np.argmax(model.predict(state, verbose=0))  

    # Perform action in environment
    state, reward, done, _, _ = env.step(action)

# Close environment
env.close()