import numpy as np
import matplotlib.pyplot as plt

# Load and plot trajectory data
trajectory = np.load("car_trajectory.npy")
x_positions, y_positions = trajectory[:, 0], trajectory[:, 1]

plt.figure(figsize=(12, 8))
plt.plot(x_positions, y_positions, label="Car Trajectory", color="blue")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Car's Trajectory on the Track")
plt.legend()
plt.grid()
plt.savefig("car_trajectory.png")
plt.show()

# Load rewards
rewards = np.load("rewards_per_episode.npy")
rewards_smoothed = np.convolve(rewards, np.ones(20)/20, mode='valid')
# Plot
plt.figure(figsize=(20, 5))
plt.plot(rewards_smoothed, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Progression Over Episodes')
plt.legend()
plt.grid()
plt.savefig("reward_progression_smoothed.png")
plt.show()
