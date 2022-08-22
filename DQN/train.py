import gym
import modules
import torch
import agents
import replay_buffer
# import time

# 性能影响可能是因为没有replay buffer导致训练数据相关性过强
# 想要再提高性能估计就得用双Q网络来改进

class TrainManager():
    
    def __init__(self, env, episodes = 2000, buffer_size = 2000, batch_size = 32, num_steps = 4,
            lr = 0.001, gamma = 0.9, epsilon = 0.1, replay_start_size = 200) :
        self.env = env
        self.episodes = episodes
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

            # 更新Q表格
            self.agent.learn(obs, action, reward, next_obs, done)

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
        for e in range(self.episodes) :
            ep_reward = self.train_episode()
            print('Episode %s: reward = %.1f'%(e, ep_reward))

            # if (e+1) % 500 == 0 :
        test_reward = self.test_episode()
        print('test_reward = %.1f'%(test_reward))


if __name__ == '__main__':
    env1 = gym.make('CartPole-v1')
    tm = TrainManager(env1)
    tm.train()