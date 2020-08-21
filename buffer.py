import numpy as np
from collections import namedtuple

np.random.seed(42)

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'not_done'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        pushed = Transition(*args)
        if len(self.memory) == self.capacity:
            self.memory[self.position] = pushed
        else:
            self.memory.append(pushed)
        self.position = (self.position + 1) % self.capacity



    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory),batch_size,replace=False)
        return np.array(self.memory)[indices]



    def __len__(self):
        return len(self.memory)
