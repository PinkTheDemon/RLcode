# from ast import main
import random
import collections
# from tempfile import NamedTemporaryFile
from torch import FloatTensor 
import numpy as np

# class buffer(Enum) :
#     obs = 0
#     action = 1
#     reward = 2
#     nextobs = 3
#     done = 4
#     prior = 5
#     no_ = 6
#     weight = 7

class ReplayBuffer() :
    def __init__(self, buffer_size, num_steps) -> None:
        self.buffer = collections.deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.num_steps = num_steps
        self.no_ = 0

    def append(self, exp) :
        # 旧的走了以后全得更新，但SumTree也得全更新啊，不然你怎么索引呢
        if len(self.buffer) < self.buffer_size :
            exp.append(self.no_)
            exp.append(1.0) # 初始权重设为1
            self.buffer.append(exp)
        else :
            # 删掉优先级最低的
            _, _, _, _, _, prior_batch, no_batch, _ = zip(*self.buffer)
            min_num = np.argmin(prior_batch)
            exp.append(no_batch[min_num])
            exp.append(1.0)
            self.buffer[no_batch[min_num]] = exp
        if self.no_ < self.buffer_size : 
            self.no_ += 1
    
    def sample(self, batch_size) :
        self.norm_prior()
        mini_batch = random.choices(self.buffer, weights = self.prob_list, k = batch_size)
        obs_batch, action_batch, reward_batch, nextobs_batch, done_batch, prior_batch, no_batch, weight_batch = zip(*mini_batch)
        obs_batch = FloatTensor(obs_batch)
        action_batch = FloatTensor(action_batch)
        reward_batch = FloatTensor(reward_batch)
        nextobs_batch = FloatTensor(nextobs_batch)
        done_batch = FloatTensor(done_batch)
        prior_batch = FloatTensor(prior_batch)
        weight_batch = FloatTensor(weight_batch)

        return obs_batch, action_batch, reward_batch, nextobs_batch, done_batch, prior_batch, no_batch, weight_batch

    def norm_prior(self) :
        sum = 0
        self.prob_list = []
        for i in range(len(self.buffer)) :
            sum += self.buffer[i][5]
        for i in range(len(self.buffer)) :
            self.prob_list.append(self.buffer[i][5] / sum)
        for i in range(len(self.buffer)) :
            self.buffer[i][7] = 1 / (len(self.buffer) * self.prob_list[i]) # 还有个beta次方先没加，看看效果再说

    def update_prior(self, prior_batch, no_batch) :
        for i in range(len(no_batch)) :
            self.buffer[no_batch[i]][5] = prior_batch[i]

    def __len__(self) :
        return len(self.buffer)

# if __name__ == '__main__' :
#     a = collections.deque(maxlen=3)
#     a.append([1,2])
#     a.append([2,3])
#     a.append([3,1])
#     sum = 0
#     for i in range(len(a)) :
#         sum += a[i][1]
#     print(sum)
#     for i in range(len(a)) :
#         a[i][1] /= sum # 存的内容是tuple形式的无法修改内容
#     print(a)
#     x, y = zip(*a) # list也是可以zip的
#     print(x)
#     # print(buffer.prior)