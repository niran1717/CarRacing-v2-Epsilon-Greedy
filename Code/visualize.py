
import gymnasium as gym
import numpy as np
import tensorflow as tf
from car_racing_dqn import CarRacingDQN  

def get_car_position(env):
    """Returns the (x, y) position of the car in the environment."""
    if hasattr(env, 'car') and hasattr(env.car, 'hull'):
        # Access car's position if available
        return env.car.hull.position
    else:
        print("Car position is not available. Ensure environment is initialized correctly.")
        return None


def visualize_model(checkpoint_path="./checkpoints"):
    # Initialize CarRacing-v2 environment with rendering
    env = gym.make("CarRacing-v2", render_mode="human")

    # Define the configuration parameters (ensure these match training config)
    config = {
            "min_epsilon": 0.1,
            "gamma": 0.95,
            "frame_skip": 3,
            "train_freq": 4,
            "batchsize": 64,
            "epsilon_decay_steps": int(1e5),
            "network_update_freq": int(1e3),
            "experience_capacity": int(4e4)
    }



    # Initialize the CarRacingDQN model
    agent = CarRacingDQN(env=env, **config)

    # Set up checkpoint for loading model weights
    checkpoint = tf.train.Checkpoint(model=agent.model)
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=5)

    # Restore the latest checkpoint if available
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Model restored from {manager.latest_checkpoint}")
    else:
        print("No checkpoint found. Please train the model first.")
        return

    # Initialize observation stack with the first frame repeated `num_frame_stack` times
    initial_obs, _ = env.reset()
    initial_obs = agent.process_image(initial_obs)
    obs_stack = [initial_obs] * agent.num_frame_stack  # Start with 4 copies of initial frame
    obs = np.stack(obs_stack, axis=0)  # Shape: (4, 96, 96)

    done = False
    total_reward = 0
    trajectory = []
    while not done:

        car_position = get_car_position(env)
        if car_position is not None:
            trajectory.append(np.array(car_position))
        else:
            print("Car position not retrieved; skipping this frame.")


        # Expand dims to match model input shape and select action
        q_values = agent.model(np.expand_dims(obs, axis=0))  # Shape: (1, 4, 96, 96)
        action_idx = np.argmax(q_values)
        action = agent.action_map[action_idx] if agent.action_map is not None else action_idx
        
        # Step the environment and accumulate rewards
        reward, terminated, truncated = 0, False, False
        for _ in range(agent.frame_skip):
            next_obs, r, terminated, truncated, _ = env.step(action)
            reward += r

            if terminated or truncated:
                break

        # Render each step
        env.render()

        # Process the next observation
        next_obs = agent.process_image(next_obs)
        
        # Update the observation stack
        obs_stack.append(next_obs)       # Add new frame to stack
        obs_stack.pop(0)                 # Remove the oldest frame
        obs = np.stack(obs_stack, axis=0)  # Shape: (4, 96, 96)
        
        total_reward += reward

        # Check if episode has ended
        if terminated or truncated:
            done = True

    trajectory = np.array([pos for pos in trajectory if pos is not None])  # Remove None values if any
    np.save("car_trajectory.npy", trajectory)
    print(f"Total Reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    visualize_model(checkpoint_path="./checkpoints")
    