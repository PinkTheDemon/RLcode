import imp
import random
import collections
from torch import FloatTensor 

class ReplayBuffer() :
    def __init__(self, buffer_size, num_steps) -> None:
        self.buffer = collections.deque(maxlen=buffer_size)
        self.num_steps = num_steps

    def append(self, exp) :
        self.buffer.append(exp)
    
    def sample(self, batch_size) :
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, nextobs_batch, done_batch = zip(*mini_batch)
        obs_batch = FloatTensor(obs_batch)
        action_batch = FloatTensor(action_batch)
        reward_batch = FloatTensor(reward_batch)
        nextobs_batch = FloatTensor(nextobs_batch)
        done_batch = FloatTensor(done_batch)

        return obs_batch, action_batch, reward_batch, nextobs_batch, done_batch

    def __len__(self) :
        return len(self.buffer)