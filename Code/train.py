import os
import numpy as np
import tensorflow as tf
import gymnasium as gym
from car_racing_dqn import CarRacingDQN

def get_car_position(env):
    """Returns the (x, y) position of the car in the environment."""
    if hasattr(env, 'car') and hasattr(env.car, 'hull'):
        # Access car's position if available
        return env.car.hull.position
    else:
        print("Car position is not available. Ensure environment is initialized correctly.")
        return None

env = gym.make("CarRacing-v2")

model_config = {
    "min_epsilon": 0.1,
    "gamma": 0.95,
    "frame_skip": 3,
    "train_freq": 4,
    "batchsize": 64,
    "epsilon_decay_steps": int(1e5),
    "network_update_freq": int(1e3),
    "experience_capacity": int(4e4)
}

dqn_agent = CarRacingDQN(env=env, **model_config)

checkpoint_path = "./checkpoints"
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint = tf.train.Checkpoint(model=dqn_agent.model, optimizer=dqn_agent.optimizer)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=5)

def save_checkpoint():
    manager.save()
    print(f"Checkpoint saved at step {dqn_agent.global_counter}")

rewards_per_episode = []
def main_loop(episodes=2000):
    for episode in range(episodes):

        reward = dqn_agent.play_episode()
        rewards_per_episode.append(reward)
        print(f"Episode {episode}, Reward: {reward}")
        if dqn_agent.global_counter % 400 == 0:
            save_checkpoint()

    # Save rewards for plotting
    np.save("rewards_per_episode.npy", rewards_per_episode)        

if __name__ == "__main__":
    main_loop()
