# IGLU Gridworld RL Environment

Fast and scalable reinforcement learning environment for the IGLU competition at NeurIPS 2022. The env represents an embodied agent with an ability to navigate, place, and break blocks of six different colors.

IGLU is a research project aimed at bridging the gap between reinforcement learning and natural language understanding in Minecraft as a collaborative environment. It provides the RL environment where the goal of an agent is to build structures within a dedicated zone. The structures are described by natural language in the game’s chat.

<img src="https://images.aicrowd.com/uploads/ckeditor/pictures/913/content_179063375-3df54656-6a72-4c73-9020-8a0f76620c28_AdobeExpress__1_.gif">

## Installation

#### Local installation

Install env:

```sh
pip install git+https://github.com/iglu-contest/gridworld.git@master
```

#### Conda:

Clone the repo and build local conda env:

```sh
git clone https://github.com/iglu-contest/gridworld.git
cd gridworld && conda env create -f env.yml
```

#### Docker:

```sh
docker build -t gridworld -f ./docker/Dockerfile .
```

## Usage:

Note that by default IGLU env runs in headless mode. To run headed do 

```sh
export IGLU_HEADLESS=0
```

Now, run the environment loop:

```python
import gym
import gridworld
from gridworld.tasks import DUMMY_TASK

# create vector based env. Rendering is enabled by default. 
# To turn it off, use render=False.
env = gym.make('IGLUGridworld-v0')

# It is mandatory to task the environemnt. 
# For dummy looping, you can use a DUMMY_TASK object:
env.set_task(DUMMY_TASK)
done = False
obs = env.reset()
while not done:
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
```

### Action spaces:

Two action spaces are available in Gridworld:

**Walking actions** allow the agent to move with gravity enabled, jump, place, break, and select blocks from inventory. 

```python
env = gym.make('IGLUGridworld-v0', action_space='walking')
print(env.action_space) # Discrete(18)
```

The action space format is the following:

  * 0 - no-op
  * 1 - step forward
  * 2 - step backward
  * 3 - step left 
  * 4 - step right
  * 5 - jump
  * 6-11 - inventory select
  * 12 - move the camera left
  * 13 - move the camera right
  * 14 - move the camera up
  * 15 - move the camera down
  * 16 - break block 
  * 17 - place block

For each movement action, the agent steps for about 0.25 of one block. Camera movement changes each angle by 5 degrees. 

**Flying actions** allow the agent to fly freely within the building zone. Placement actions are the same and movement is specified by a continuous vector.

```python
env = gym.make('IGLUGridworld-v0', action_space='flying')
```

Action space format:

```python
Dict(
  movement: Box(low=-1, high=1, shape=(3,)),
  camera: Box(low=-5, high=5, shape=(2,)),
  inventory: Discrete(7),
  placement: Discrete(3)
)
```

### Observation space

Observation space format:

```python
Dict(
  inventory: Box(low=0, high=20, shape=(6,)),
  compass: Box(low=-180, high=180, shape=(1,)),
  dialog: String(),
  pov: Box(low=0, high=255, shape=(64, 64, 3))
)
```

Here, `inventory` indicates the total blocks available to the agent (per color).
The `compass` component shows the angle between the agent's yaw angle and the North direction.
The `dialog` value is the full previous dialog and the most recent instruction to execute.
Finally, `pov` is an ego-centric image of the agent's observation of the world.
Note that **this space will be used during the evaluation.**
However, it is possible to access other fields of the environment, for example, during training.
The `vector_state=True` passed as keyword argument in `gym.make` will return, in addition to previous fields,

```python
agentPos: Box(low=[-8, -2, -8, -90, 0], high=[8, 12, 8, 90, 360], shape=(5,)),
grid: Box(low=-1, high=7, shape=(9, 11, 11))
```

It is also possible to make a target grid a part of the observation space. To do that, pass `target_in_obs=True` to `gym.make`. This will add another key to the observation space with the same structure as the `grid` component. The name of a new component is `target_grid`. This part of the space remains fixed within an episode.

### Reward calculation

Each step, the reward is calculated based on the similarity between the so far built grid and the target grid. The reward is determined regardless of global spatial position of currently placed blocks, it only takes into account how much the built blocks are similar to the target structure. To make it possible, at each step we calculate the intersection between the built and the target structures for each spatial translation within the horizontal plane and rotation around the vertical axis. Then we take the maximal intersection value among all translation and rotations. To calculate the reward, we compare the maximal intersection size from the current step with the one from the previous step. We reward the agent with `2` for the increase of the maximal intersection size, with `-2` for the decrease of the maximal intersection size, and with `1`/`-1` for removing/placing a block without a change of the maximal intersection size. A visual example is shown below.

<img src="./assets/intersections.png" width="256">

Specifically, we run the code that is equivalent to the following one:

```python
def maximal_intersection(grid, target_grid):
  """
  Args:
    grid (np.ndarray[Y, X, Z]): numpy array snapshot of a built structure
    target_grid (np.ndarray[Y, X, Z]): numpy array snapshot of the target structure
  """
  maximum = 0
  # iterate over orthogonal rotations
  for i in range(4):
    # iterate over translations
    for dx in range(-X, X + 1):
      for dz in range(-Z, Z + 1):
        shifted_grid = translate(grid, dx, dz)
        intersection = np.sum( (shifted_grid == target) & (target != 0) )
        maximum = max(maximum, intersection)
    grid = rotate_y_axis(grid)
  return maximum
```

In practice, a more optimized version is used. The reward is then calculated based on the temporal difference between maximal intersection of the two consecutive grids. Formally, suppose `grids[t]` is a built structure at timestep `t`. The reward is then calculated as:

```python
def calc_reward(prev_grid, grid, target_grid, , right_scale=2, wrong_scale=1):
  prev_max_int = maximal_intersection(prev_grid, target_grid)
  max_int = maximal_intersection(grid, target_grid)
  diff = max_int - prev_max_int
  prev_grid_size = num_blocks(prev_grid)
  grid_size = num_blocks(grid)
  if diff == 0:
    return wrong_scale * np.sign(grid_size - prev_grid_size)
  else:
    return right_scale * np.sign(diff)
```

In other words, if a recently placed block strictly increases or decreases the maximal intersection, the reward is positive or negative and is equal to `+/-right_scale`. Otherwise, its absolute value is equal to `wrong_scale` and the sign is positive if a block was removed or negative if added.
Values `right_scale` and `wrong_scale` can be passed to `gym.make` as environment kwargs. Finally, the `maximal_intersection` includes heavy computations that slow down the environment. They can be simplified by disabling rotational/translational invariance at the cost of much more sparse reward. To do that, pass `invariant=False` to a corresponding `Task` object (see Dataset section for reference).


## Working with the IGLU dataset 

![test](https://raw.githubusercontent.com/iglu-contest/gridworld/aicrowd-launch-prep/assets/c118_1_step_0.mp4)

By default, the environment requires a task object to run.
IGLU dataset provides a convenient loader for RL tasks. Here is an example of how to use it:

```python
import gym
from gridworld.data import IGLUDataset

dataset = IGLUDataset(dataset_version='v0.1.0-rc1') 
# leave dataset_version empty to access the most recent version of the dataset.

env = gym.make('IGLUGridworld-v0')
env.set_task_generator(dataset)
```

In this example, we download the dataset of tasks for RL env. 
Internally, on each `.reset()` of the env, the dataset samples a random task (inside its own `.reset()` method) and makes it active in the env. The `Task` object is responsible for calculating the reward, providing the text part of the observation, and determining if the episode has ended.

The structure of the IGLU dataset is following. The dataset consists of structures that represent overall collaboration goals. For each structure, we have several collaboration sessions that pair architects with builders to build each particular structure. Each session consists of a sequence of "turns". Each turn represents an *atomic* instruction and corresponding changes of the blocks in the world. The structure of a `Task` object is following:

  * `target_grid` - target blocks configuration that needs to be built
  * `starting_grid` - optional, blocks for the environment to begin the episode with.
  * `chat` - full conversation between the architect and builder, including the most recent instruction
  * `last_instruction` - last utterance of the architect

Sometimes, the instructions can be ambiguous and the builder asks a clarifying question which the architect answers. In the latter case, `last_instruction` will contain three utterances: an instruction, a clarifying question, and an answer to that question. Otherwise, `last_instruction` is just one utterance of the architect.

To represent collaboration sessions, the `Subtasks` class is used. This class represents a sequence of dialog utterances and their corresponding goals (each of which is a partially completed structure). On `.reset()` call, it picks a random turn and returns a `Task` object, where starting and target grids are consecutive partial structures and the dialog contains all utterances up until the one corresponding to the target grid.

In the example above, the dataset object is structured as follows:

```python
# .tasks is a dict mapping from structure to a list of sessions of interaction
dataset.tasks 
# each value contains a list corresponding to collaboration sessions.
dataset.tasks['c73']
# Each element of this list is an instance of `Subtasks` class
dataset.tasks['c73'][0]
```

The `.reset()` method of `IGLUDataset` does effectively the following:

```python
def reset(dataset):
  task_id = random.choice(dataset.tasks.keys())
  session = random.choice(dataset.tasks[task_id])
  subtask = session.reset() # Task object is returned
  return subtask
```

This behavior can be customized simply by overriding the reset method in a subclass:

```python
import gym
from gridworld.data import IGLUDataset

class MyDataset(IGLUDataset):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.my_task_id = 'c73'
    self.my_session = 0
  
  def reset(self):
    return self.tasks[self.my_task_id][self.my_session].reset()

env = gym.make('IGLUGridworld-v0')
my_dataset = MyDataset(dataset_version='v0.1.0-rc1')
env.set_task_generator(my_dataset)
# do training/sampling
```

On the first creation, the dataset is downloaded and parsed automatically. Below you will find the structure of the dataset:

```
dialogs.csv
builder-data/
  ...
  1-c118/ # session id - structure_id
    step-2
  ...
  9-c118/
    step-2
    step-4
    step-6
  1-c120/
    step-2
  ...
  23-c126/
    step-2
    step-4
    step-6
    step-8
```

Here, `dialog.csv` contains the utterances of architects and builders solving different tasks in 
different sessions. The `builder-data/` directory contains builder behavior recorded by the voxel.js engine. Right now we extract only the resulting grids and use them as targets.

## Known Issues
