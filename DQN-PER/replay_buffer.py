from ast import main
import imp
import random
import collections
from tempfile import NamedTemporaryFile
from torch import FloatTensor 

class ReplayBuffer() :
    def __init__(self, buffer_size, num_steps) -> None:
        self.buffer = collections.deque(maxlen=buffer_size)
        self.num_steps = num_steps

    def append(self, exp) :
        self.buffer.append(exp)
    
    def sample(self, batch_size) :
        self.norm_prior()
        mini_batch = random.choices(self.buffer, weights = self.prob_list, k = batch_size)
        obs_batch, action_batch, reward_batch, nextobs_batch, done_batch, prior_batch = zip(*mini_batch)
        obs_batch = FloatTensor(obs_batch)
        action_batch = FloatTensor(action_batch)
        reward_batch = FloatTensor(reward_batch)
        nextobs_batch = FloatTensor(nextobs_batch)
        done_batch = FloatTensor(done_batch)

        return obs_batch, action_batch, reward_batch, nextobs_batch, done_batch, prior_batch

    def norm_prior(self) :
        sum = 0
        for i in range(len(self.buffer)) :
            sum += self.buffer[i][5]
        for i in range(len(self.buffer)) :
            self.prob_list[i] = self.buffer[i][5] / sum

    def __len__(self) :
        return len(self.buffer)

# if __name__ == '__main__' :
#     a = collections.deque(maxlen=3)
#     a.append([1,2])
#     a.append([2,3])
#     a.append([3,1])
#     print(a)
#     sum = 0
#     for i in range(len(a)) :
#         sum += a[i][1]
#     print(sum)
#     for i in range(len(a)) :
#         a[i][1] /= sum # 存的内容是tuple形式的无法修改内容
#     print(a)
#     x, y = zip(*a) # list也是可以zip的
#     print(x)