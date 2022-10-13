import gym
# import modules
import torch
import agents
import replay_buffer
import matplotlib.pyplot as plt
import time
import math
from turtle import forward


USE_CUDA = torch.cuda.is_available()
#将变量放到cuda上
Variable = lambda *args, **kwargs: torch.autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else torch.autograd.Variable(*args, **kwargs)

class NoiseLinear(torch.nn.Module) :
    def __init__(self, in_features, out_features, std_init = 0.4) : 
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = torch.nn.parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = torch.nn.parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = torch.nn.parameter(torch.FloatTensor(out_features))
        self.bias_sigma = torch.nn.parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_paremeters()
        self.reset_noise()

    def forward(self, x) :
        if self.training :
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else :
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.uniform_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class MLP(torch.nn.Module) : 
    # ->none 和 super 起什么作用？
    # none表示输出为空，super是跟继承有关的
    def __init__(self, obs_size, n_act) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(obs_size, 50)
        self.linear2 = torch.nn.Linear(50, 50)
        self.linear3 = torch.nn.Linear(50, 50)
        self.noise = NoiseLinear(50,n_act)

    def forward(self, x) :
        x = torch.nn.functional.relu(self.linear1)
        x = torch.nn.functional.relu(self.linear2)
        x = torch.nn.functional.relu(self.linear1)
        x = self.noise

        return x
    
    def reset_noise(self) :
        self.noise.reset_noise()

# 性能影响可能是因为没有replay buffer导致训练数据相关性过强
# 想要再提高性能估计就得用双Q网络来改进
# 双Q网络改进并不明显，而且统一的问题是训练过程抖动很大，探索出了高奖励值的动作并没有保持下来
# 好像说replay buffer直接丢掉最老的经验是不太好的做法，因为这样训练到后面智能体会忘掉环境刚开始的经验
# 可能需要按优先级来排序然后删

class TrainManager():
    
    def __init__(self, env, episodes = 3000, buffer_size = 10000, batch_size = 128, num_steps = 4, e_decay_episode = 1,
            lr = 0.0001, gamma = 0.99, epsilon = 0.5, replay_start_size = 256, update_target_step = 20, epsilon_decay = 0.0002) :
        self.env = env
        self.episodes = episodes
        self.epsilon = epsilon
        self.e_decay_episode = e_decay_episode
        self.epsilon_decay = epsilon_decay
        n_obs = env.observation_space.shape[0]
        n_act = env.action_space.n
        q_func = MLP(n_obs, n_act)
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

            # # 尝试修改奖励值看能不能提升训练性能
            # if reward == 0 :
            #     reward = 10000
            # # 结果是不行的

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
            '''/epsilon decay'''

            # '''plot'''
            e_list.append(e)
            reward_list.append(ep_reward)
            # plt.xlim(max(0, e-200), e)
            # plt.plot(e_list, reward_list, color = "black", ls = '-')
            # plt.pause(0.1)
            # '''/plot'''

        time_end = time.time()
        print('time = %.2f'%(time_end-time_start))
        while (int(input())) :
            test_reward = self.test_episode()
            print('test_reward = %.1f'%(test_reward))
            plt.xlim(0, e)
            plt.plot(e_list, reward_list, color = "black", ls = '-')
            continue

if __name__ == '__main__':
    time_start = time.time()
    env1 = gym.make('CartPole-v1')
    tm = TrainManager(env1)
    tm.train()