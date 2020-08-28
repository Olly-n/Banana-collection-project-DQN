from sum_tree import SumTree
from collections import deque, namedtuple
import numpy as np
import random


class PrioritisedReplayBuffer():
    """A prioritised replay buffer.

    Creates a sum tree and uses it to stores a fixed number of experience tuples. When sampled
    experiences are returned with greater priority given to those with the highest absolute TD-error.
    """

    def __init__(self, buffer_size, alpha, beta_zero, beta_increment_size=0.001, epsilon = 0.1, max_priority=1., seed = None):
        """Priority replay buffer initialiser.

        Args:
            buffer_size (int): capacity of the replay buffer.
            alpha (float): priority scaling hyperparameter.
            beta_zero (float): importance sampling scaling hyperparameter.
            beta_increment_size (float): beta annealing rate.
            epsilon (float): base priority to ensure non-zero sampling probability.
            max_priority (float): initial maximum priority.
            seed (int): seed for random number generator
       """
        random.seed(seed)

        self.sum_tree = SumTree(buffer_size)
        self.memory = {}
        self.experience = namedtuple("experience", ["state", "action", "reward", "next_state", "done"])
        self.buffer_size = buffer_size
        self.beta_increment_size = beta_increment_size
        self.max_priority = max_priority**alpha
        self.min_priority = max_priority**alpha
        self.last_min_update = 0

        self.alpha = alpha
        self.beta = beta_zero
        self.epsilon = epsilon

    def add(self, state, action, reward, next_state, done):
        """Creates experience tuple and adds it to the replay buffer."""
        experience = self.experience(state, action, reward, next_state, done)
        current_tree_idx = self.sum_tree.input_pointer
        self.memory[current_tree_idx] = experience
        self.sum_tree.add(self.max_priority)

    def sample(self, batch_size):
        """Returns a batch of experiences sampled according to their priority."""
        idx_list = []
        weights = []
        states = []
        actions = []
        rewards = []
        next_states = []
        done_list = []

        segment = self.sum_tree.total()/batch_size
        sample_list = [random.uniform(segment*i, segment*(i+1)) for i in range(batch_size)]
        max_weight = self.min_priority**(-self.beta)

        for s in sample_list:
            idx, priority = self.sum_tree.sample(s)
            idx_list.append(idx)
            weight = priority**(-self.beta) / max_weight
            weights.append(weight)

            sample = self.memory[idx]
            state, action, reward, next_state, done = sample
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            done_list.append(done)

        return states, actions, rewards, next_states, done_list, idx_list, weights

    def update(self, idx_list, td_error):
        """Updates a specifics experience's priority."""
        priority_list = (td_error+self.epsilon)**self.alpha


        self.max_priority = max(self.max_priority, priority_list.max())
        list_min_priority = priority_list.min()

        if list_min_priority <= self.min_priority:
            self.min_priority = list_min_priority
            self.last_min_update = 0
        else:
            self.last_min_update += 1

        if self.last_min_update >= self.buffer_size:
            self.min_priority = np.array([node.val for node in self.sum_tree.tree_array[-self.buffer_size:]]).min()
            self.last_min_update = 0

        for i, idx in enumerate(idx_list):
            priority = min(self.max_priority, priority_list[i])
            self.sum_tree.update(idx, priority)

        self.beta = min(1, self.beta + self.beta_increment_size)

    def __len__(self,):
        """Return number of experiences in the replay buffer."""
        return len(self.memory)


class ReplayBuffer():
    """A replay buffer.

    Used to store a fixed number of experience tuples obtained while interacting
    with the environment.
    """

    def __init__(self, buffer_size, seed):
        """Creates a replay buffer.

        Args:
            buffer_size (int): Maximum size of the replay buffer.
            seed (int): Random seed used for batch selection.
        """
        self.memory = deque(maxlen=buffer_size)
        self.seed = random.seed(seed)
        self.experience = namedtuple("experience", ["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Creates experience tuple and adds it to the replay buffer."""
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size):
        """Returns a sampled batch of experiences."""
        sampled_batch = random.sample(self.memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        done_list = []

        for sample in sampled_batch:
            state, action, reward, next_state, done = sample
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            done_list.append(done)

        return states, actions, rewards, next_states, done_list

    def __len__(self):
        """Return number of experiences in the replay buffer."""
        return len(self.memory)
