# dqn.py
import gymnasium as gym
import numpy as np
from skimage import color, transform
import tensorflow as tf
from experience_history import ExperienceHistory

class DQN:
    def __init__(self, env, batchsize=64, pic_size=(96, 96), num_frame_stack=4,
                 gamma=0.95, frame_skip=1, train_freq=4, initial_epsilon=1.0,
                 min_epsilon=0.1, render=True, epsilon_decay_steps=int(1e6),
                 min_experience_size=int(1e3), experience_capacity=int(1e5),
                 network_update_freq=5000, regularization=1e-6,
                 optimizer_params=None, action_map=None):
        

        self.min_epsilon = min_epsilon
        self.initial_epsilon = initial_epsilon
        self.min_experience_size = min_experience_size
        self.frame_skip = frame_skip
        self.regularization = regularization
        self.num_frame_stack = num_frame_stack
        self.exp_history = ExperienceHistory(num_frame_stack, experience_capacity, pic_size)
        self.playing_cache = ExperienceHistory(num_frame_stack, num_frame_stack * 5 + 10, pic_size)
        self.dim_actions = len(action_map) if action_map is not None else env.action_space.n
        self.network_update_freq = network_update_freq
        self.env = env
        self.action_map = action_map
        self.batchsize = batchsize
        self.gamma = gamma
        self.render = render
        self.train_freq = train_freq
        self.epsilon_decay_steps = epsilon_decay_steps
        self.pic_size = pic_size
        self.optimizer_params = optimizer_params or dict(learning_rate=0.0004, epsilon=1e-7)
        
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(**self.optimizer_params)
        
        self.global_counter = 0
        self.episode_counter = 0
        self.do_training = True

    def process_image(self, img):
        return 2 * color.rgb2gray(transform.resize(img[34:194], (96, 96))) - 1

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=(self.num_frame_stack, *self.pic_size))
        x = tf.keras.layers.Permute((2, 3, 1))(inputs)
        x = tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation="relu")(x)
        x = tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation="relu")(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        outputs = tf.keras.layers.Dense(self.dim_actions)(x)
        return tf.keras.Model(inputs, outputs)

    def get_epsilon(self):
        if not self.do_training:
            return 0.1
        decay = max(self.min_epsilon, self.initial_epsilon - self.global_counter / self.epsilon_decay_steps)
        return decay

    def train_step(self, batch):
        states = batch["prev_state"]
        actions = batch["actions"]
        rewards = batch["reward"]
        next_states = batch["next_state"]
        dones = batch["done_mask"]
        next_q_values = tf.reduce_max(self.target_model(next_states), axis=1)
        q_targets = rewards + (1 - dones) * self.gamma * next_q_values
        
        with tf.GradientTape() as tape:
            q_values = tf.reduce_sum(self.model(states) * tf.one_hot(actions, self.dim_actions), axis=1)
            loss = tf.reduce_mean(tf.square(q_targets - q_values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    

    def play_episode(self):
        
        """Runs one episode."""
        obs, _ = self.env.reset()
        obs = self.process_image(obs)
        
        self.exp_history.start_new_episode(obs)
        
        total_reward = 0
        terminated, truncated = False, False
        while not (terminated or truncated):
            epsilon = self.get_epsilon()
            if np.random.rand() < epsilon:
                action_idx = self.get_random_action()
            else:
                action_idx = np.argmax(self.model(np.expand_dims(self.exp_history.current_state(), axis=0)))
            
            action = self.action_map[action_idx] if self.action_map is not None else action_idx

            reward, done = 0, False
            trajectory = []
            for _ in range(self.frame_skip):
                next_obs, r, terminated, truncated, _ = self.env.step(action)
                reward += r
                if terminated or truncated:
                    break        

            next_obs = self.process_image(next_obs)
            self.exp_history.add_experience(next_obs, action_idx, done, reward)
            total_reward += reward
            if self.do_training:
                self.global_counter += 1
                if self.global_counter % self.train_freq == 0:
                    batch = self.exp_history.sample_mini_batch(self.batchsize)
                    self.train_step(batch)
                if self.global_counter % self.network_update_freq == 0:
                    self.target_model.set_weights(self.model.get_weights())
            
            if done:
                break
        
      
        self.exp_history.end_episode()
        return total_reward
