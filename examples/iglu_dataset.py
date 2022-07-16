from gridworld.data import IGLUDataset
import gridworld
import gym
import sys
sys.path.insert(0, '../')

DATASET_VERSION = "v0.1.0-rc1"

# enable per task translation-invariant reward
iglu_dataset = IGLUDataset(
    dataset_version=DATASET_VERSION, task_kwargs={'invariant': False})
print(f'total structures: {len(iglu_dataset.tasks)}')
print(f'total sessions: {len(sum(iglu_dataset.tasks.values(), []))}')
print(
    f'total total RL tasks: {sum(len(sess.structure_seq) for sess in sum(iglu_dataset.tasks.values(), []))}')

env = gym.make('IGLUGridworldVector-v0')
env.set_task_generator(iglu_dataset)

obs = env.reset()
# get task info
print('Dataset sample:\n******')
print(env.task.chat)
print('******')
print('Dataset sample:\n******')
obs = env.reset()
# here should be different task
print(env.task.chat)
print('******')

# drop task generator and set and individual task
env.set_task_generator(None)
# without line above, the .set_task will have no effect
# you should call .set_tasks_generator(None) only after
# setting task generator to any non trivial value
print('Task sample:\n******')
env.set_task(iglu_dataset.tasks['c118'][0])
env.reset()
print(env.task.chat)
print('******')

# interaction with this env will happen such that once the sub-goal is
# reached, the env internally switches to a new goal until the last one is solved.
# turn off this behavior, use progressive=False
iglu_dataset = IGLUDataset(dataset_version=DATASET_VERSION, task_kwargs={
                           'invariant': False, 'progressive': False})

# object under iglu_dataset.tasks['c118'][0] is an instance
# of Subtasks which itself is a task generator
# (it outputs subsequent segments within the task).
subtask = iglu_dataset.tasks['c118'][0].reset()
env.set_task(subtask)
env.reset()
