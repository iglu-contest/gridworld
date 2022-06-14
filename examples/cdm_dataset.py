import gym
import gridworld
from gridworld.data import CDMDataset

# enable per task translation-invariant reward
cdm_dataset = CDMDataset(task_kwargs={'invariant': False}) 
print(f'total structures: {len(cdm_dataset.tasks)}')
print(f'total sessions (and RL tasks): {len(sum(cdm_dataset.tasks.values(), []))}')

env = gym.make('IGLUGridworldVector-v0')
env.set_task_generator(cdm_dataset)

obs = env.reset()
# get task info
print(env.task.chat)
obs = env.reset()
# here should be different task
print(env.task.chat)

# drop task generator and set and individual task
env.set_task_generator(None)
# without line above, the .set_task will have no effect
# you should call .set_tasks_generator(None) only after 
# setting task generator to any non trivial value
env.set_task(cdm_dataset.tasks['c10'][0])
env.reset()
print(env.task.chat)

# object under iglu_dataset.tasks['c118'][0] is an instance
# of Subtasks which itself is a task generator 
# (it outputs subsequent segments within the task).