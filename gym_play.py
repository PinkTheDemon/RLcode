import gym
import random 

env = gym.make("CliffWalking-v0")

state = env.reset()
while True :
    # action = random.randint(0,3)
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    env.render()
    if done :
        break