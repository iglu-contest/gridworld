import gym
import gridworld
from gridworld.tasks import DUMMY_TASK

# create vector based env. Rendering is enabled by default. 
# To turn it off, use render=False.
# Note that without 
env = gym.make('IGLUGridworldVector-v0')

env.set_task(DUMMY_TASK)
print(f'Action space: {env.action_space}')
print(f'Observation space: {env.observation_space}')
done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

