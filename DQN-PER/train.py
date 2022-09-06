import gym
import modules
import torch
import agents
import replay_buffer
import matplotlib.pyplot as plt
# import time

# 性能影响可能是因为没有replay buffer导致训练数据相关性过强
# 想要再提高性能估计就得用双Q网络来改进
# 双Q网络改进并不明显，而且统一的问题是训练过程抖动很大，探索出了高奖励值的动作并没有保持下来
# 好像说replay buffer直接丢掉最老的经验是不太好的做法，因为这样训练到后面智能体会忘掉环境刚开始的经验
# 可能需要按优先级来排序然后删
# 用了PER之后，它所需要的附加操作真的非常耗时


class TrainManager():
    
    def __init__(self, env, episodes = 3000, buffer_size = 5000, batch_size = 32, num_steps = 4, e_decay_episode = 1,
            lr = 0.0001, gamma = 0.99, epsilon = 1.0, replay_start_size = 100, update_target_step = 32, epsilon_decay = 0.001) :
        self.env = env
        self.episodes = episodes
        self.epsilon = epsilon
        self.e_decay_episode = e_decay_episode
        self.epsilon_decay = epsilon_decay
        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n
        q_func = modules.MLP(n_obs, n_act)
        optimizer = torch.optim.AdamW(q_func.parameters(), lr=lr)
        rb = replay_buffer.ReplayBuffer(buffer_size, num_steps)
        self.agent = agents.DQNAgent(
            q_func=q_func,
            optimizer=optimizer,
            replay_buffer = rb,
            batch_size = batch_size,
            replay_start_size = replay_start_size,
            n_act=n_act,
            update_target_step=update_target_step,
            gamma=gamma,
            epsilon=epsilon
        )

    # 训练一轮游戏
    def train_episode(self):
        total_reward = 0 # 总奖励值
        obs = self.env.reset() # 初始化环境

        while True :
            action = self.agent.act(obs) # 根据下一状态预测下一动作
            next_obs, reward, done, _ = self.env.step(action) # 前进一步
            prior = 1 # 新采集的样本优先级设为最大

            # 更新Q表格
            self.agent.learn(obs, action, reward, next_obs, done, prior)

            obs = next_obs
            total_reward += reward

            if done : break

        return total_reward

    def test_episode(self):
        total_reward = 0
        obs = self.env.reset() # 初始化环境

        while True :
            action = self.agent.predict(obs) # 测试不需要探索
            obs, reward, done, _ = self.env.step(action)

            total_reward += reward
            self.env.render()
            # time.sleep(0.5) # 0.5秒运行一步

            if done : break

        return total_reward


    def train(self) :
        plt.ion()
        plt.figure(1)
        e_list = []
        reward_list = []

        for e in range(self.episodes) :
            ep_reward = self.train_episode()
            print('Episode %s: reward = %.1f'%(e, ep_reward))
            '''epsilon decay'''
            if e % self.e_decay_episode == 0 :
                self.epsilon -= self.epsilon_decay
                self.epsilon = max(0.005, self.epsilon)
                self.agent.epsilon_decay(self.epsilon)
            # '''plot'''
            e_list.append(e)
            reward_list.append(ep_reward)
            # plt.xlim(max(0, e-200), e)
            # plt.plot(e_list, reward_list, color = "black", ls = '-')
            # plt.pause(0.1)
            # '''/plot'''

        while (int(input())) :
            test_reward = self.test_episode()
            print('test_reward = %.1f'%(test_reward))
            plt.xlim(0, e)
            plt.plot(e_list, reward_list, color = "black", ls = '-')
            continue

if __name__ == '__main__':
    env1 = gym.make('CartPole-v1')
    tm = TrainManager(env1)
    tm.train()