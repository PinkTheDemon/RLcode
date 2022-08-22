import numpy as np 
import gym
import time

# 状态空间为0到多少的整数，动作空间也是0到多少的整数
# 所以有些写法就没管状态动作和数字的区别

class QlearningAgent() :
    def __init__(self, n_state, n_act, epsilon = 0.1, lr = 0.1, gamma = 0.9) :
        self.n_state = n_state # 状态空间大小
        self.n_act = n_act # 动作空间大小
        self.epsilon = epsilon # ε贪心参数
        self.Q = np.zeros((n_state, n_act)) # Q表格
        self.lr = lr # 学习率
        self.gamma = gamma # 收益衰减率

    def predict(self, state) : # 利用
        Q_list = self.Q[state,:] # 当前状态的各个动作的Q值列表
        # 若Q值列表中价值最大的动作有多个，则随机选取而非单纯选择标号最小的动作
        action = np.random.choice(np.flatnonzero(Q_list==Q_list.max()))
        return action

    def act(self, state) :
        if np.random.uniform(0,1) < self.epsilon : # 探索
            # 动作空间中随机选择一个动作
            action = np.random.choice(self.n_act)
        else : # 利用
            # 选择当前状态对应Q值最大的动作
            action = self.predict(state)
        return action

    def learn(self, state, action, reward, next_state, done) :
        # Q learning公式，跟sarsa的if else结构效果是一样的
        target = reward + (1 - float(done)) * self.gamma * self.Q[next_state, :].max()
        self.Q[state, action] += self.lr * (target - self.Q[state, action])

    def pirntQ(self) :
        print(self.Q)

def train_episode(env, agent):
    total_reward = 0 # 总奖励值
    state = env.reset() # 初始化环境

    while True :
        action = agent.act(state) # 根据下一状态预测下一动作
        next_state, reward, done, _ = env.step(action) # 前进一步

        # 更新Q表格
        agent.learn(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done : break

    return total_reward

def test_episode(env, agent):
    total_reward = 0
    state = env.reset()

    while True :
        action = agent.predict(state) # 测试不需要探索
        state, reward, done, _ = env.step(action)

        total_reward += reward
        env.render()
        time.sleep(0.5) # 0.5秒运行一步

        if done : break

    return total_reward


def train(env, episodes = 500, lr = 0.1, gamma = 0.9, epsilon = 0.1) :
    agent = QlearningAgent(
        n_state = env.observation_space.n,
        n_act=env.action_space.n,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon
    )

    for e in range(episodes) :
        ep_reward = train_episode(env, agent)
        print('Episode %s: reward = %.1f'%(e, ep_reward))
    
    test_reward = test_episode(env,agent)
    print('test_reward = %.1f'%(test_reward))
    agent.pirntQ()

if __name__ == '__main__':
    env = gym.make('CliffWalking-v0')
    train(env)