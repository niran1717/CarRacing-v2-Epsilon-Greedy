# car_racing_dqn.py
import numpy as np
import itertools as it
from dqn import DQN

class CarRacingDQN(DQN):
    def __init__(self, env, max_negative_rewards=100, **kwargs):
        all_actions = np.array(list(it.product([-1, 0, 1], [1, 0], [0.2, 0])))
        kwargs["action_map"] = all_actions
        kwargs["pic_size"] = (96, 96)
        kwargs["render"] = True
        super().__init__(env= env, **kwargs)

        self.gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in all_actions])
        self.break_actions = np.array([a[2] == 1 for a in all_actions])
        self.n_gas_actions = self.gas_actions.sum()
        self.neg_reward_counter = 0
        self.max_neg_rewards = max_negative_rewards

    
    def get_random_action(self):
        action_weights = 14.0 * self.gas_actions + 1.0
        action_weights /= np.sum(action_weights)
        
        # Randomly choose an action with the specified probabilities
        return np.random.choice(self.dim_actions, p=action_weights)
        
    def check_early_stop(self, reward, totalreward):
        if reward < 0:
            self.neg_reward_counter += 1
            done = (self.neg_reward_counter > self.max_neg_rewards)

            if done and totalreward <= 500:
                punishment = -20.0
            else:
                punishment = 0.0
            if done:
                self.neg_reward_counter = 0

            return done, punishment
        else:
            self.neg_reward_counter = 0
            return False, 0.0
