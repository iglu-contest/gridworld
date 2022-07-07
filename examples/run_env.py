import sys
sys.path.insert(0, '../')
import gym
import gridworld
from gridworld.tasks import DUMMY_TASK
from time import perf_counter

# create vector based env. Rendering is enabled by default. 
# To turn it off, use render=False.
# Note that without 
env = gym.make('IGLUGridworldVector-v0')

env.set_task(DUMMY_TASK)
print(f'Action space: {env.action_space}')
# print(f'Observation space: {env.observation_space}')
time = 0
steps = 0
for ep in range(500):
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        t = perf_counter()
        obs, reward, done, info = env.step(action)
        print(obs["grid"].shape, reward)
        time += perf_counter() - t
        steps += 1
print(f'steps per second: {steps / time:.4f}')
