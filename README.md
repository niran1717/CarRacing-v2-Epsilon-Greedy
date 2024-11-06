# CarRacing-v2-Epsilon-Greedy
# Reinforcement Learning for Autonomous Car Control using Epsilon-Greedy Policy

This project demonstrates a reinforcement learning approach to autonomous car control in a simulated environment using the Gymnasium `CarRacing-v2` environment. The project uses a Deep Q-Network (DQN) architecture and employs an epsilon-greedy exploration policy to balance exploration and exploitation during training.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Results](#results)
- [Future Work](#future-work)

## Project Overview
The goal of this project is to train an agent to drive autonomously in a simulated environment by optimizing its driving behavior using reinforcement learning. An epsilon-greedy policy is implemented to explore actions and update the policy based on rewards obtained from the environment.

### Epsilon-Greedy Policy
The epsilon-greedy policy allows the agent to explore actions randomly with a probability epsilon (ε), promoting exploration in early stages of training. As training progresses, epsilon decays, focusing the agent on exploiting learned strategies.

## Project Architecture
1. **Experience Replay**: Stores past experiences to reduce correlation in data and improve model stability.
2. **DQN Model**: A neural network that approximates Q-values for actions, with convolutional layers for feature extraction.
3. **Training Loop**: Manages the episodic training, where the agent iteratively learns optimal actions.
4. **CarRacing Environment**: A simulated car racing environment with continuous control actions for steering, acceleration, and braking.

## Installation
Clone this repository and navigate to the project directory

## Usage
### Create a virtual environment (recommended):
python3 -m venv venv

### Install the required packages:
pip install -r requirements.txt

### To train the agent, run the following command:
python train.py

### To visualize the agent’s behavior in the CarRacing-v2 environment, use:
python visualize.py

## Files
- experience_history.py: Implements the Experience Replay Buffer.
- dqn.py: Contains the DQN class with the neural network architecture.
- car_racing_dqn.py: Customizes the DQN for the car racing environment.
- train.py: Manages the training loop and checkpoints.
- visualize.py: Visualizes the agent's performance after training.
- plot.py: plots the required graphs or results

## Results
Throughout training, the agent's rewards improve as it learns to navigate the environment. Epsilon-greedy exploration allows the agent to find an optimal driving strategy by balancing exploration of new actions with exploitation of known strategies.

## Future Work
Possible enhancements include:
- Implementing adaptive epsilon decay for more dynamic exploration.
- Experimenting with continuous control strategies, such as DDPG.
- Applying transfer learning to reduce training time by leveraging prior experience in similar environments.

