import gym
import gridworld
from gridworld.tasks import DUMMY_TASK

env = gym.make('IGLUGridworldVector-v0')
env.set_task(DUMMY_TASK)

done = False
obs = env.reset()
print(f'Action space: {env.action_space}')
print(f'Observation space: {env.observation_space}')

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
