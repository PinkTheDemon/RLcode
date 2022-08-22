import numpy as np 
import torch
import torchUtils
import copy

# 状态空间为0到多少的整数，动作空间也是0到多少的整数
# 所以有些写法就没管状态动作和数字的区别

class DQNAgent() :
    def __init__(self, q_func, optimizer, n_act, replay_buffer, batch_size, replay_start_size, update_target_step, epsilon, gamma) :
        self.q_func = q_func # 动作价值函数
        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss() # 损失函数

        self.globle_step = 0
        self.rb = replay_buffer
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

        self.n_act = n_act # 动作空间大小
        self.epsilon = epsilon # ε贪心参数
        self.targetQ = copy.deepcopy(q_func) # 目标网络
        self.gamma = gamma # 收益衰减率
        self.update_target_step = update_target_step # 目标网络更新率

    def predict(self, obs) : # 利用
        obs = torch.FloatTensor(obs)
        Q_list = self.q_func(obs) # 
        # 若Q值列表中价值最大的动作有多个，则随机选取而非单纯选择标号最小的动作
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    def predict_batch(self, obs_batch) :
        Q_list = self.q_func(obs_batch)
        action_batch = Q_list.max(1).indices
        return action_batch

    def act(self, obs) :
        if np.random.uniform(0,1) < self.epsilon : # 探索
            # 动作空间中随机选择一个动作
            action = np.random.choice(self.n_act)
        else : # 利用
            # 选择当前状态对应Q值最大的动作
            action = self.predict(obs)
        return action

    def learn_batch(self, obs_batch, action_batch, reward_batch, next_obs_batch, done_batch) :
        # 双Q网络，当前网络选择最大的动作，目标网络求该动作对应的动作价值
        pred_Vs = self.q_func(obs_batch)
        action_onehot = torchUtils.one_hot(action_batch, self.n_act)
        predictQ = (pred_Vs*action_onehot).sum(dim=1)
        nextaction_batch = self.predict_batch(next_obs_batch)
        nextaction_onehot = torchUtils.one_hot(nextaction_batch, self.n_act)
        targetQ = reward_batch + (1 - done_batch) * self.gamma * (self.targetQ(next_obs_batch) * nextaction_onehot).sum(1)
        
        # 更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(predictQ, targetQ)
        loss.backward()
        self.optimizer.step()

    def learn(self, obs, action, reward, next_obs, done) :
        self.globle_step += 1
        self.rb.append((obs, action, reward, next_obs, done))
        if self.rb.__len__() > self.replay_start_size  and self.globle_step%self.rb.num_steps == 0 :
            self.learn_batch(*self.rb.sample(self.batch_size))

        if self.globle_step % self.update_target_step == 0 :
            self.updateTargetQ()

    def updateTargetQ(self) :
        for target_param, param in zip(self.targetQ.parameters(), self.q_func.parameters()) :
            target_param.data.copy_(param.data)